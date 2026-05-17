import importlib
import sys

from intake import create_analysis_run
from models import AgentFindings, MemoOutput, StageResult
from orchestrator import PipelineResult


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeStreamlit:
    def __init__(self):
        self.session_state = {}
        self.errors: list[str] = []

    def set_page_config(self, **kwargs):
        pass

    def error(self, message):
        self.errors.append(message)

    def stop(self):
        raise AssertionError("st.stop should not be called during test setup")

    def title(self, *args, **kwargs):
        pass

    def subheader(self, *args, **kwargs):
        pass

    def markdown(self, *args, **kwargs):
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

    def button(self, *args, **kwargs):
        return False

    def status(self, *args, **kwargs):
        return _NullContext()

    def write(self, *args, **kwargs):
        pass

    def caption(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def expander(self, *args, **kwargs):
        return _NullContext()

    def rerun(self):
        pass


def _import_app(monkeypatch):
    import config

    fake_st = FakeStreamlit()
    monkeypatch.setattr(config, "get_settings", lambda: object())
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.delitem(sys.modules, "app", raising=False)
    return importlib.import_module("app"), fake_st


def _make_stage_results() -> list[StageResult]:
    findings = AgentFindings.model_construct(
        sources=["https://example.com/market"],
        confidence=0.8,
        key_findings=["Growing TAM"],
        evidence_gaps=[],
    )
    return [StageResult(stage_name="market", status="completed", findings=findings)]


def _make_memo() -> MemoOutput:
    findings = AgentFindings.model_construct(
        sources=["https://example.com/market"],
        confidence=0.8,
        key_findings=["Growing TAM"],
        evidence_gaps=[],
    )
    return MemoOutput.model_construct(
        executive_summary="Promising market with manageable diligence gaps.",
        research_findings={"market": findings},
        independent_review=None,
        recommendation="Watch",
        confidence=0.65,
        confidence_factors=["All 4 research perspectives completed"],
        unresolved_risks=["Customer concentration risk"],
        open_questions=["What is gross margin?"],
        sources=["https://example.com/market"],
    )


def test_finalize_run_stores_memo_on_analysis_run(monkeypatch, sample_run_input):
    app, fake_st = _import_app(monkeypatch)
    fake_st.session_state["analysis_run"] = create_analysis_run(sample_run_input)
    saved_runs = []
    memo = _make_memo()

    monkeypatch.setattr(app, "save_run", lambda run: saved_runs.append(run))

    app._finalize_run(
        PipelineResult(
            status="complete",
            stage_results=_make_stage_results(),
            memo=memo,
        )
    )

    updated = fake_st.session_state["analysis_run"]
    assert len(saved_runs) == 1
    assert updated.memo == memo
    assert saved_runs[0].memo == memo


def test_finalize_run_stores_none_when_pipeline_result_has_no_memo(monkeypatch, sample_run_input):
    app, fake_st = _import_app(monkeypatch)
    fake_st.session_state["analysis_run"] = create_analysis_run(sample_run_input)
    saved_runs = []

    monkeypatch.setattr(app, "save_run", lambda run: saved_runs.append(run))

    app._finalize_run(
        PipelineResult(
            status="partial",
            stage_results=_make_stage_results(),
            memo=None,
        )
    )

    updated = fake_st.session_state["analysis_run"]
    assert updated.status == "partial"
    assert updated.memo is None
    assert saved_runs[0].memo is None
