import importlib
import sys

from fake_streamlit import FakeStreamlit

from models import AgentFindings, AnalysisRun, MemoOutput, RunInput, StageResult
from orchestrator import PipelineResult


def _import_app(monkeypatch, session_state=None):
    import config
    import persistence

    fake_st = FakeStreamlit()
    if session_state:
        fake_st.session_state.update(session_state)
    monkeypatch.setattr(config, "get_settings", lambda: object())
    monkeypatch.setattr(persistence, "list_runs", lambda **kwargs: [])
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.delitem(sys.modules, "app", raising=False)
    return importlib.import_module("app"), fake_st


def _make_complete_memo() -> MemoOutput:
    findings = AgentFindings.model_construct(
        sources=["https://example.com/market"],
        confidence=0.8,
        key_findings=["Growing TAM"],
        evidence_gaps=[],
    )
    return MemoOutput.model_construct(
        executive_summary="Promising early-stage company with strong market positioning.",
        research_findings={"market": findings},
        independent_review=None,
        recommendation="Watch",
        confidence=0.65,
        confidence_factors=["All 4 research perspectives completed", "Independent Review completed"],
        unresolved_risks=["Customer concentration risk", "Unclear unit economics"],
        open_questions=["What is the 12-month retention profile?"],
        sources=["https://example.com/market", "https://example.com/competition"],
    )


def _make_low_confidence_memo() -> MemoOutput:
    memo = _make_complete_memo()
    return MemoOutput.model_construct(**{**memo.__dict__, "confidence": 0.35})


def _make_empty_list_memo() -> MemoOutput:
    memo = _make_complete_memo()
    return MemoOutput.model_construct(
        **{
            **memo.__dict__,
            "confidence_factors": [],
            "unresolved_risks": [],
            "open_questions": [],
            "sources": [],
        }
    )


def _make_stage_results() -> list[StageResult]:
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


def _make_analysis_run(memo: MemoOutput | None) -> AnalysisRun:
    return AnalysisRun.model_construct(
        id="run-1",
        status="complete",
        created_at="2026-06-04T00:00:00+00:00",
        input=RunInput.model_construct(startup_name="Acme Robotics"),
        stage_results=_make_stage_results(),
        memo=memo,
    )


def _make_pipeline_result(memo: MemoOutput | None) -> PipelineResult:
    return PipelineResult(status="complete", stage_results=_make_stage_results(), memo=memo)


def test_render_memo_display_renders_summary_recommendation_and_confidence(monkeypatch):
    app, fake_st = _import_app(monkeypatch)

    app.render_memo_display(_make_complete_memo(), "complete")

    assert "Promising early-stage company with strong market positioning." in fake_st.markdowns
    assert "### Recommendation: Watch" in fake_st.markdowns
    assert "**Confidence:** 65%" in fake_st.markdowns


def test_render_memo_display_creates_collapsible_audit_sections(monkeypatch):
    app, fake_st = _import_app(monkeypatch)
    fake_st.expander_labels.clear()

    app.render_memo_display(_make_complete_memo(), "complete")

    assert fake_st.expander_labels == [
        "Confidence Factors",
        "Unresolved Risks — 2 items",
        "Open Questions — 1 item",
        "Sources — 2 references",
    ]


def test_render_memo_display_shows_empty_list_audit_fallbacks(monkeypatch):
    app, fake_st = _import_app(monkeypatch)

    app.render_memo_display(_make_empty_list_memo(), "complete")

    assert fake_st.markdowns.count("None identified") == 4


def test_render_memo_display_shows_disclaimer_banner(monkeypatch):
    app, fake_st = _import_app(monkeypatch)
    fake_st.infos.clear()

    app.render_memo_display(_make_complete_memo(), "complete")

    assert fake_st.infos == [
        "⚠️ This memo is a decision-support artifact based on public information. "
        "It is not investment advice. All findings require independent verification "
        "and human judgment."
    ]


def test_render_memo_display_shows_uncertainty_caption(monkeypatch):
    app, fake_st = _import_app(monkeypatch)

    app.render_memo_display(_make_complete_memo(), "complete")

    assert fake_st.captions == [
        "Confidence reflects evidence strength across completed research perspectives, "
        "adjusted for contradictions, gaps, and missing data."
    ]


def test_partial_run_shows_incomplete_evidence_warning(monkeypatch):
    app, fake_st = _import_app(monkeypatch)

    app.render_memo_display(_make_complete_memo(), "partial")

    assert (
        "⚠️ This recommendation is based on incomplete evidence. "
        "Some research stages failed or did not complete. "
        "Review confidence factors below for details."
    ) in fake_st.warnings


def test_low_confidence_warning_appears_below_threshold(monkeypatch):
    app, fake_st = _import_app(monkeypatch)

    app.render_memo_display(_make_low_confidence_memo(), "complete")

    assert "Low confidence score — significant evidence gaps or contradictions were identified." in fake_st.warnings


def test_results_flow_renders_memo_when_analysis_run_has_memo(monkeypatch):
    memo = _make_complete_memo()

    _app, fake_st = _import_app(
        monkeypatch,
        {
            "analysis_run": _make_analysis_run(memo),
            "pipeline_progress": {
                "stage_results": _make_stage_results(),
                "pipeline_result": _make_pipeline_result(memo),
            },
        },
    )

    assert "Decision Memo" in fake_st.subheaders


def test_results_flow_skips_memo_when_analysis_run_memo_is_none(monkeypatch):
    _app, fake_st = _import_app(
        monkeypatch,
        {
            "analysis_run": _make_analysis_run(None),
            "pipeline_progress": {
                "stage_results": _make_stage_results(),
                "pipeline_result": _make_pipeline_result(None),
            },
        },
    )

    assert "Decision Memo" not in fake_st.subheaders
