import logging

import streamlit as st

from config import ConfigError, get_settings
from intake import build_run_input, create_analysis_run
from models import RunInput
from persistence import save_run

logger = logging.getLogger(__name__)
RUN_FEEDBACK_KEY = "analysis_run_feedback"


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


def render_summary_field(label: str, value: str | None) -> None:
    if value is None:
        return
    st.markdown(f"**{label}:**")
    st.text(value)


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
    st.info("A run is already in progress.")
