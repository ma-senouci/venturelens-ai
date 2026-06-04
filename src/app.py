import logging
import time

import streamlit as st

from config import ConfigError, get_settings
from intake import build_run_input, create_analysis_run
from models import AgentFindings, MemoOutput, RunInput, StageResult
from orchestrator import CRITIC_STAGE, PIPELINE_STAGES, PipelineResult
from persistence import save_run
from pipeline_runner import start_pipeline_thread

logger = logging.getLogger(__name__)
RUN_FEEDBACK_KEY = "analysis_run_feedback"

STAGE_LABELS = {
    "market": "Market Research",
    "competition": "Competition Analysis",
    "product": "Product Positioning",
    "risk": "Risk Assessment",
    CRITIC_STAGE: "Independent Review",
}
STATUS_ICONS = {"completed": "✅", "failed": "❌", "in_progress": "⏳", "pending": "⬜"}

_POLL_INTERVAL_SECONDS = 0.5
_DISCLAIMER_TEXT = (
    "⚠️ This memo is a decision-support artifact based on public information. "
    "It is not investment advice. All findings require independent verification "
    "and human judgment."
)
_CONFIDENCE_CAPTION = (
    "Confidence reflects evidence strength across completed research perspectives, "
    "adjusted for contradictions, gaps, and missing data."
)
_PARTIAL_RUN_WARNING = (
    "⚠️ This recommendation is based on incomplete evidence. "
    "Some research stages failed or did not complete. "
    "Review confidence factors below for details."
)
_LOW_CONFIDENCE_WARNING = "Low confidence score — significant evidence gaps or contradictions were identified."


def start_analysis_run(run_input: RunInput) -> None:
    analysis_run = create_analysis_run(run_input)
    try:
        save_run(analysis_run)
    except Exception:
        logger.exception("Failed to persist run %s", analysis_run.id)
        st.session_state[RUN_FEEDBACK_KEY] = (
            "error",
            "Failed to save the analysis run. Please try again.",
        )
    else:
        st.session_state["analysis_run"] = analysis_run
        st.session_state[RUN_FEEDBACK_KEY] = (
            "success",
            f"Analysis run started for {analysis_run.input.startup_name}.",
        )
        logger.info(
            "Analysis run started for '%s' (run_id=%s, input_id=%s)",
            analysis_run.input.startup_name,
            analysis_run.id,
            analysis_run.input.id,
        )

        settings = get_settings()
        progress = {"stage_results": [], "pipeline_result": None}
        st.session_state["pipeline_progress"] = progress
        thread = start_pipeline_thread(run_input, settings, progress)
        st.session_state["pipeline_thread"] = thread


def render_summary_field(label: str, value: str | None) -> None:
    if value is None:
        return
    st.markdown(f"**{label}:**")
    st.text(value)


def _find_stage_result(stage_results: list[StageResult], stage_name: str) -> StageResult | None:
    for sr in stage_results:
        if sr.stage_name == stage_name:
            return sr
    return None


def _is_completed_research_stage(stage_result: StageResult) -> bool:
    return (
        stage_result.stage_name in PIPELINE_STAGES
        and stage_result.status == "completed"
        and isinstance(stage_result.findings, AgentFindings)
    )


def render_progress_display() -> None:
    progress = st.session_state.get("pipeline_progress")
    if progress is None:
        return

    stage_results = progress["stage_results"]
    pipeline_result = progress["pipeline_result"]
    pipeline_finished = pipeline_result is not None

    if pipeline_finished:
        if pipeline_result.status == "complete":
            label, state, expanded = "Analysis complete", "complete", False
        elif pipeline_result.status == "partial":
            label, state, expanded = "Analysis partially complete", "error", True
        else:
            label, state, expanded = "Analysis failed", "error", True
    else:
        label, state, expanded = "Running analysis...", "running", True

    progress_stage_names = list(PIPELINE_STAGES)
    critic_stage_result = _find_stage_result(stage_results, CRITIC_STAGE)
    if critic_stage_result is not None or not pipeline_finished:
        progress_stage_names.append(CRITIC_STAGE)

    with st.status(label, expanded=expanded, state=state):
        for stage_name in progress_stage_names:
            stage_result = _find_stage_result(stage_results, stage_name)
            if stage_result:
                icon = STATUS_ICONS[stage_result.status]
                st.write(f"{icon} {STAGE_LABELS[stage_name]}")
                if stage_result.error:
                    st.caption(f"Error: {stage_result.error}")
            else:
                st.write(f"{STATUS_ICONS['pending']} {STAGE_LABELS[stage_name]}")

    if pipeline_finished and pipeline_result.status == "partial":
        st.warning("Some research stages failed. Results shown are from completed stages only.")

    if pipeline_finished:
        _finalize_run(pipeline_result)


def _finalize_run(pipeline_result: PipelineResult) -> None:
    analysis_run = st.session_state.get("analysis_run")
    if analysis_run is None or analysis_run.status != "running":
        return

    updated = analysis_run.model_copy(
        update={
            "status": pipeline_result.status,
            "stage_results": pipeline_result.stage_results,
            "memo": pipeline_result.memo,
        }
    )
    st.session_state["analysis_run"] = updated

    try:
        save_run(updated)
    except Exception:
        logger.exception("Failed to persist completed run %s", updated.id)
        st.error("Failed to save the completed analysis run.")


def render_findings_display(stage_results: list[StageResult]) -> None:
    has_completed = any(_is_completed_research_stage(sr) for sr in stage_results)
    if not has_completed:
        return

    st.subheader("Research Findings")

    for sr in stage_results:
        if sr.stage_name not in PIPELINE_STAGES:
            continue

        label = STAGE_LABELS.get(sr.stage_name, sr.stage_name)

        if sr.status == "failed":
            st.caption(f"❌ {label} — Failed: {sr.error}")
            continue

        if not _is_completed_research_stage(sr):
            continue

        findings = sr.findings
        finding_count = len(findings.key_findings)
        with st.expander(f"{label} — ✅ {finding_count} findings", expanded=False):
            if findings.key_findings:
                st.markdown("**Key Findings:**")
                for i, finding in enumerate(findings.key_findings, 1):
                    st.markdown(f"{i}. {finding}")

            if findings.sources:
                st.markdown("**Sources for this section:**")
                for source in findings.sources:
                    st.markdown(f"- 🔗 {source}")

            if findings.evidence_gaps:
                st.warning("**Evidence Gaps:**\n" + "\n".join(f"- {gap}" for gap in findings.evidence_gaps))


def _collect_all_sources(stage_results: list[StageResult]) -> list[str]:
    seen: dict[str, bool] = {}
    for sr in stage_results:
        if _is_completed_research_stage(sr):
            for source in sr.findings.sources:
                stripped = source.strip()
                if stripped and stripped not in seen:
                    seen[stripped] = True
    return list(seen.keys())


def render_consolidated_sources(stage_results: list[StageResult]) -> None:
    completed_research_stage_names = {sr.stage_name for sr in stage_results if _is_completed_research_stage(sr)}
    if completed_research_stage_names != set(PIPELINE_STAGES):
        return

    all_sources = _collect_all_sources(stage_results)
    if not all_sources:
        return

    with st.expander(f"Consolidated Sources — {len(all_sources)} references", expanded=False):
        for i, source in enumerate(all_sources, 1):
            st.markdown(f"{i}. {source}")


def _render_bulleted_items(items: list[str]) -> None:
    if not items:
        st.markdown("None identified")
        return

    for item in items:
        st.markdown(f"- {item}")


def render_memo_display(memo: MemoOutput, run_status: str) -> None:
    st.subheader("Decision Memo")
    st.info(_DISCLAIMER_TEXT)
    if run_status == "partial":
        st.warning(_PARTIAL_RUN_WARNING)
    st.markdown(memo.executive_summary)
    st.markdown(f"### Recommendation: {memo.recommendation}")
    st.markdown(f"**Confidence:** {memo.confidence:.0%}")
    if memo.confidence < 0.5:
        st.warning(_LOW_CONFIDENCE_WARNING)
    st.caption(_CONFIDENCE_CAPTION)

    with st.expander("Confidence Factors", expanded=False):
        _render_bulleted_items(memo.confidence_factors)

    n_risks = len(memo.unresolved_risks)
    risks_label = f"Unresolved Risks — {n_risks} item{'s' if n_risks != 1 else ''}"
    with st.expander(risks_label, expanded=False):
        _render_bulleted_items(memo.unresolved_risks)

    questions_label = f"Open Questions — {len(memo.open_questions)} item{'s' if len(memo.open_questions) != 1 else ''}"
    with st.expander(questions_label, expanded=False):
        _render_bulleted_items(memo.open_questions)

    sources_label = f"Sources — {len(memo.sources)} reference{'s' if len(memo.sources) != 1 else ''}"
    with st.expander(sources_label, expanded=False):
        if not memo.sources:
            st.markdown("None identified")
        else:
            for i, source in enumerate(memo.sources, 1):
                st.markdown(f"{i}. {source}")


st.set_page_config(
    page_title="VentureLens AI",
    page_icon="🔍",
    layout="wide",
)

try:
    get_settings()
except ConfigError as error:
    st.error(str(error))
    st.stop()

run_feedback = st.session_state.pop(RUN_FEEDBACK_KEY, None)
has_active_run = "analysis_run" in st.session_state

st.title("🔍 VentureLens AI")
st.subheader("AI-Powered Startup Due Diligence")
st.markdown("Enter a startup name and any supporting context to prepare a structured, source-backed diligence run.")
if run_feedback is not None:
    level, message = run_feedback
    if level == "success":
        st.success(message)
    else:
        st.error(message)

with st.form("intake_form"):
    startup_name = st.text_input(
        "Startup name",
        placeholder="e.g. Acme Robotics",
        key="startup_name",
    )
    website_url = st.text_input(
        "Company website",
        placeholder="e.g. https://acmerobotics.com",
        key="website_url",
    )
    description = st.text_area(
        "What does the startup do?",
        placeholder="Summarize the product, customer, and problem it solves in 1-2 sentences",
        key="description",
    )
    thesis = st.text_area(
        "Why is this startup worth evaluating?",
        placeholder="Add your thesis, angle, or reasons this company may be promising",
        key="thesis",
    )
    analysis_focus = st.text_input(
        "What should the analysis focus on?",
        placeholder="e.g. market size, moat durability, team strength, technical risk",
        key="analysis_focus",
    )
    submitted = st.form_submit_button(
        "Prepare analysis",
        use_container_width=True,
        disabled=has_active_run,
    )

if submitted:
    st.session_state.pop("run_input", None)
    try:
        run_input = build_run_input(
            startup_name=startup_name,
            website_url=website_url,
            description=description,
            thesis=thesis,
            analysis_focus=analysis_focus,
        )
        st.session_state["run_input"] = run_input
        st.success(f"Analysis inputs prepared for {run_input.startup_name}.")
        logger.info(
            "Analysis inputs prepared for '%s' (id=%s)",
            run_input.startup_name,
            run_input.id,
        )
    except ValueError as exc:
        logger.warning("Intake validation failed: %s", exc)
        st.error(str(exc))

display_input: RunInput | None = None
if has_active_run:
    analysis_run = st.session_state["analysis_run"]
    display_input = analysis_run.input
elif "run_input" in st.session_state:
    display_input = st.session_state["run_input"]

if display_input is not None:
    st.subheader("Analysis inputs")
    render_summary_field("Startup", display_input.startup_name)
    render_summary_field("Website", display_input.website_url)
    render_summary_field("Description", display_input.description)
    render_summary_field("Thesis", display_input.thesis)
    render_summary_field("Analysis focus", display_input.analysis_focus)
    st.button(
        "Run analysis",
        use_container_width=True,
        disabled=has_active_run,
        on_click=start_analysis_run,
        args=(display_input,),
    )

if has_active_run:
    render_progress_display()

    progress = st.session_state.get("pipeline_progress")
    pipeline_result = progress.get("pipeline_result") if progress else None
    if pipeline_result is not None:
        analysis_run = st.session_state.get("analysis_run")
        if analysis_run and analysis_run.stage_results:
            render_findings_display(analysis_run.stage_results)
            if pipeline_result.status == "complete":
                render_consolidated_sources(analysis_run.stage_results)
            if analysis_run.memo is not None:
                render_memo_display(analysis_run.memo, analysis_run.status)

    thread = st.session_state.get("pipeline_thread")
    if thread and thread.is_alive():
        time.sleep(_POLL_INTERVAL_SECONDS)
        st.rerun()
