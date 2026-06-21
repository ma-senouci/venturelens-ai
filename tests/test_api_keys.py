import importlib
import sys

from fake_streamlit import FakeStreamlit


def _import_app(monkeypatch, session_state=None, env_keys=None):
    import config
    import persistence

    fake_st = FakeStreamlit()
    if session_state:
        fake_st.session_state.update(session_state)

    if env_keys:
        for env_var, value in env_keys.items():
            monkeypatch.setenv(env_var, value)

    if env_keys:
        monkeypatch.setattr(config, "get_settings", config.get_settings)
    else:
        monkeypatch.setattr(config, "get_settings", lambda: (_ for _ in ()).throw(config.ConfigError("missing")))

    monkeypatch.setattr(persistence, "list_runs", lambda **kwargs: [])
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.delitem(sys.modules, "app", raising=False)
    return importlib.import_module("app"), fake_st


def test_resolve_settings_with_sidebar_keys_returns_valid_settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)

    from config import get_settings as real_get_settings

    app, fake_st = _import_app(
        monkeypatch,
        session_state={
            "sidebar_openai_api_key": "sk-from-sidebar",
            "sidebar_serper_api_key": "serper-from-sidebar",
        },
    )

    monkeypatch.setattr(app, "get_settings", real_get_settings)

    result = app._resolve_settings()

    assert result is not None
    assert result.openai_api_key == "sk-from-sidebar"
    assert result.serper_api_key == "serper-from-sidebar"


def test_resolve_settings_with_no_keys_returns_none(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)

    app, _fake_st = _import_app(monkeypatch)

    result = app._resolve_settings()

    assert result is None


def test_resolve_settings_with_env_vars_only_returns_valid_settings(monkeypatch):
    app, _fake_st = _import_app(
        monkeypatch,
        env_keys={"OPENAI_API_KEY": "sk-env", "SERPER_API_KEY": "serper-env"},
    )

    result = app._resolve_settings()

    assert result is not None
    assert result.openai_api_key == "sk-env"
    assert result.serper_api_key == "serper-env"


def test_sidebar_renders_api_configuration_with_password_inputs(monkeypatch):
    _app, fake_st = _import_app(monkeypatch)

    api_inputs = [inp for inp in fake_st.sidebar_text_inputs if inp["type"] == "password"]

    assert len(api_inputs) == 2
    labels = [inp["label"] for inp in api_inputs]
    assert "OpenAI API Key" in labels
    assert "Serper API Key" in labels


def test_intake_form_disabled_when_no_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)

    _app, fake_st = _import_app(monkeypatch)

    submit_buttons = [b for b in fake_st.form_submit_buttons if b["label"] == "Prepare analysis"]
    assert len(submit_buttons) == 1
    assert submit_buttons[0]["disabled"] is True


def test_intake_form_enabled_when_keys_provided(monkeypatch):
    _app, fake_st = _import_app(
        monkeypatch,
        env_keys={"OPENAI_API_KEY": "sk-test", "SERPER_API_KEY": "serper-test"},
    )

    submit_buttons = [b for b in fake_st.form_submit_buttons if b["label"] == "Prepare analysis"]
    assert len(submit_buttons) == 1
    assert submit_buttons[0]["disabled"] is False
