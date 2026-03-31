import re
from pathlib import Path

import pytest

from config import DEFAULT_OPENAI_MODEL_NAME, ConfigError, get_settings

_TEST_WORKDIR = Path(__file__).resolve().parent / "_config_test_workdir"


def _set_required_env(monkeypatch, *, openai_api_key="sk-test", serper_api_key="serper-test") -> None:
    monkeypatch.setenv("OPENAI_API_KEY", openai_api_key)
    monkeypatch.setenv("SERPER_API_KEY", serper_api_key)


@pytest.fixture(autouse=True)
def reset_settings_cache(monkeypatch):
    _TEST_WORKDIR.mkdir(exist_ok=True)
    monkeypatch.chdir(_TEST_WORKDIR)
    for env_var in ("OPENAI_API_KEY", "OPENAI_MODEL_NAME", "SERPER_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_get_settings_loads_required_values_from_environment(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-5-mini")

    settings = get_settings()

    assert settings.openai_api_key == "sk-test"
    assert settings.openai_model_name == "gpt-5-mini"
    assert settings.serper_api_key == "serper-test"


def test_get_settings_uses_default_openai_model_name(monkeypatch):
    _set_required_env(monkeypatch)

    settings = get_settings()

    assert settings.openai_model_name == DEFAULT_OPENAI_MODEL_NAME


def test_get_settings_reports_missing_openai_api_key(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "serper-test")

    with pytest.raises(ConfigError, match=re.escape("Missing required environment variable: OPENAI_API_KEY")):
        get_settings()


def test_get_settings_reports_missing_serper_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with pytest.raises(ConfigError, match=re.escape("Missing required environment variable: SERPER_API_KEY")):
        get_settings()


def test_get_settings_reports_multiple_missing_variables_in_deterministic_order():
    with pytest.raises(
        ConfigError,
        match=re.escape("Missing required environment variables: OPENAI_API_KEY, SERPER_API_KEY"),
    ):
        get_settings()


def test_get_settings_treats_empty_required_values_as_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("SERPER_API_KEY", "serper-test")

    with pytest.raises(ConfigError, match=re.escape("Missing required environment variable: OPENAI_API_KEY")):
        get_settings()


def test_get_settings_rejects_whitespace_only_required_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    monkeypatch.setenv("SERPER_API_KEY", "serper-test")

    with pytest.raises(ConfigError, match=re.escape("Missing required environment variable: OPENAI_API_KEY")):
        get_settings()
