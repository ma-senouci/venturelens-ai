import importlib
import sys

from models import AgentFindings, AnalysisRun, MemoOutput, RunInput, StageResult
from persistence import RunSummary


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeSidebarContext:
    def __init__(self, parent):
        self._parent = parent

    def __enter__(self):
        self._parent._in_sidebar = True
        return self

    def __exit__(self, *args):
        self._parent._in_sidebar = False
        return False

    def header(self, text, *args, **kwargs):
        self._parent.sidebar_headers.append(text)

    def caption(self, text, *args, **kwargs):
        self._parent.sidebar_captions.append(text)


class FakeStreamlit:
    def __init__(self):
        self.session_state = {}
        self.errors: list[str] = []
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.captions: list[str] = []
        self.subheaders: list[str] = []
        self.markdowns: list[str] = []
        self.expander_labels: list[str] = []
        self.sidebar_headers: list[str] = []
        self.sidebar_captions: list[str] = []
        self.sidebar_buttons: list[dict] = []
        self._in_sidebar = False
        self._sidebar_ctx = _FakeSidebarContext(self)

    @property
    def sidebar(self):
        return self._sidebar_ctx

    def set_page_config(self, **kwargs):
        pass

    def error(self, message):
        self.errors.append(message)

    def stop(self):
        raise AssertionError("st.stop should not be called during test setup")

    def title(self, *args, **kwargs):
        pass

    def subheader(self, text, *args, **kwargs):
        self.subheaders.append(text)

    def markdown(self, text, *args, **kwargs):
        self.markdowns.append(text)

    def text(self, *args, **kwargs):
        pass

    def success(self, *args, **kwargs):
        pass

    def form(self, *args, **kwargs):
        return _NullContext()

    def text_input(self, *args, **kwargs):
        return ""

    def text_area(self, *args, **kwargs):
        return ""

    def form_submit_button(self, *args, **kwargs):
        return False

    def button(self, label="", *args, **kwargs):
        if self._in_sidebar:
            self.sidebar_buttons.append(
                {
                    "label": label,
                    "key": kwargs.get("key"),
                    "disabled": kwargs.get("disabled", False),
                }
            )
            return False
        return False

    def status(self, *args, **kwargs):
        return _NullContext()

    def write(self, *args, **kwargs):
        pass

    def caption(self, message, *args, **kwargs):
        if self._in_sidebar:
            self.sidebar_captions.append(message)
        else:
            self.captions.append(message)

    def header(self, text, *args, **kwargs):
        if self._in_sidebar:
            self.sidebar_headers.append(text)

    def warning(self, message, *args, **kwargs):
        self.warnings.append(message)

    def info(self, message, *args, **kwargs):
        self.infos.append(message)

    def expander(self, label, *args, **kwargs):
        self.expander_labels.append(label)
        return _NullContext()

    def rerun(self):
        pass

    def _clear_sidebar(self):
        self.sidebar_headers.clear()
        self.sidebar_captions.clear()
        self.sidebar_buttons.clear()


def _import_app(monkeypatch, session_state=None, list_runs_result=None):
    import config
    import persistence

    fake_st = FakeStreamlit()
    if session_state:
        fake_st.session_state.update(session_state)
    monkeypatch.setattr(config, "get_settings", lambda: object())

    if list_runs_result is not None:
        monkeypatch.setattr(persistence, "list_runs", lambda **kwargs: list_runs_result)
    else:
        monkeypatch.setattr(persistence, "list_runs", lambda **kwargs: [])

    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.delitem(sys.modules, "app", raising=False)
    return importlib.import_module("app"), fake_st


def _make_run_summary(
    run_id="run-1",
    startup_name="Acme Robotics",
    status="complete",
    created_at="2026-06-04T19:09:00+02:00",
):
    return RunSummary(id=run_id, startup_name=startup_name, status=status, created_at=created_at)


def _make_complete_memo():
    findings = AgentFindings.model_construct(
        sources=["https://example.com/market"],
        confidence=0.8,
        key_findings=["Growing TAM"],
        evidence_gaps=[],
    )
    return MemoOutput.model_construct(
        executive_summary="Promising early-stage company.",
        research_findings={"market": findings},
        independent_review=None,
        recommendation="Watch",
        confidence=0.65,
        confidence_factors=["All 4 research perspectives completed"],
        unresolved_risks=["Customer concentration risk"],
        open_questions=["What is retention?"],
        sources=["https://example.com/market"],
    )


def _make_stage_results():
    findings = AgentFindings.model_construct(
        sources=["https://example.com/market"],
        confidence=0.8,
        key_findings=["Growing TAM"],
        evidence_gaps=[],
    )
    return [
        StageResult.model_construct(stage_name="market", status="completed", findings=findings),
        StageResult.model_construct(stage_name="competition", status="completed", findings=findings),
        StageResult.model_construct(stage_name="product", status="completed", findings=findings),
        StageResult.model_construct(stage_name="risk", status="completed", findings=findings),
    ]


def _make_analysis_run(run_id="run-1", status="complete", memo=None):
    return AnalysisRun.model_construct(
        id=run_id,
        status=status,
        created_at="2026-06-04T19:09:00+02:00",
        input=RunInput.model_construct(
            id="input-1",
            startup_name="Acme Robotics",
            website_url="https://acmerobotics.com",
        ),
        stage_results=_make_stage_results(),
        memo=memo,
    )


def test_sidebar_shows_run_history_when_runs_exist(monkeypatch):
    summary = _make_run_summary()
    app, fake_st = _import_app(monkeypatch, list_runs_result=[summary])
    fake_st._clear_sidebar()

    app.render_run_history_sidebar()

    assert "Run History" in fake_st.sidebar_headers
    assert len(fake_st.sidebar_buttons) == 1
    assert "Acme Robotics" in fake_st.sidebar_buttons[0]["label"]


def test_sidebar_shows_empty_state_when_no_runs(monkeypatch):
    app, fake_st = _import_app(monkeypatch, list_runs_result=[])
    fake_st._clear_sidebar()

    app.render_run_history_sidebar()

    assert "Run History" in fake_st.sidebar_headers
    assert "No completed runs yet" in fake_st.sidebar_captions


def test_active_running_run_is_disabled_in_sidebar(monkeypatch):
    active_run = _make_analysis_run(run_id="run-1", status="running")
    summary = _make_run_summary(run_id="run-1", status="running")

    app, fake_st = _import_app(
        monkeypatch,
        session_state={"analysis_run": active_run},
        list_runs_result=[summary],
    )
    fake_st._clear_sidebar()

    app.render_run_history_sidebar()

    assert fake_st.sidebar_buttons[0]["disabled"] is True
    assert "🔄" in fake_st.sidebar_buttons[0]["label"]


def test_active_completed_run_is_selectable_in_sidebar(monkeypatch):
    active_run = _make_analysis_run(run_id="run-1", status="complete")
    summary = _make_run_summary(run_id="run-1", status="complete")

    app, fake_st = _import_app(
        monkeypatch,
        session_state={"analysis_run": active_run},
        list_runs_result=[summary],
    )
    fake_st._clear_sidebar()

    app.render_run_history_sidebar()

    assert fake_st.sidebar_buttons[0]["disabled"] is False


def test_sidebar_handles_list_runs_error(monkeypatch):
    app, fake_st = _import_app(monkeypatch)
    fake_st._clear_sidebar()

    def _fail(**kwargs):
        raise RuntimeError("DB unavailable")

    monkeypatch.setattr(app, "list_runs", _fail)

    app.render_run_history_sidebar()

    assert "Run History" in fake_st.sidebar_headers
    assert "Unable to load run history" in fake_st.sidebar_captions


def test_load_historical_run_error_sets_error_state(monkeypatch):
    def _fail_load(run_id, **kwargs):
        raise RuntimeError("corrupt data")

    app, fake_st = _import_app(monkeypatch)
    monkeypatch.setattr(app, "load_run", _fail_load)

    app._load_historical_run("run-bad")

    assert fake_st.session_state.get("historical_run_error") == "Failed to load the selected run."


def test_load_historical_run_not_found_sets_error_state(monkeypatch):
    app, fake_st = _import_app(monkeypatch)
    monkeypatch.setattr(app, "load_run", lambda run_id, **kwargs: None)

    app._load_historical_run("run-missing")

    assert fake_st.session_state.get("historical_run_error") == "Run not found."


def test_load_historical_run_stores_run_in_session_state(monkeypatch):
    historical_run = _make_analysis_run(run_id="run-hist")
    app, fake_st = _import_app(monkeypatch)
    monkeypatch.setattr(app, "load_run", lambda run_id, **kwargs: historical_run)

    app._load_historical_run("run-hist")

    assert fake_st.session_state["selected_historical_run"] is historical_run
    assert "historical_run_error" not in fake_st.session_state


def test_historical_run_renders_memo_and_findings(monkeypatch):
    memo = _make_complete_memo()
    historical_run = _make_analysis_run(run_id="run-hist", status="complete", memo=memo)

    _app, fake_st = _import_app(
        monkeypatch,
        session_state={"selected_historical_run": historical_run},
    )

    assert "Historical Run: Acme Robotics" in fake_st.subheaders
    assert "Decision Memo" in fake_st.subheaders


def test_historical_run_not_shown_when_active_run_exists(monkeypatch):
    memo = _make_complete_memo()
    historical_run = _make_analysis_run(run_id="run-hist", status="complete", memo=memo)
    active_run = _make_analysis_run(run_id="run-active", status="running")

    _app, fake_st = _import_app(
        monkeypatch,
        session_state={
            "selected_historical_run": historical_run,
            "analysis_run": active_run,
        },
    )

    assert "Historical Run: Acme Robotics" not in fake_st.subheaders


def test_start_analysis_run_clears_selected_historical_run(monkeypatch, sample_run_input):
    historical_run = _make_analysis_run(run_id="run-hist")
    app, fake_st = _import_app(
        monkeypatch,
        session_state={"selected_historical_run": historical_run},
    )

    monkeypatch.setattr(app, "save_run", lambda run: None)
    monkeypatch.setattr(app, "start_pipeline_thread", lambda *args: type("T", (), {"is_alive": lambda self: False})())

    app.start_analysis_run(sample_run_input)

    assert "selected_historical_run" not in fake_st.session_state
