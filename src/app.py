import logging

import streamlit as st

from config import ConfigError, get_settings
from intake import build_run_input

logger = logging.getLogger(__name__)

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

st.title("🔍 VentureLens AI")
st.subheader("AI-Powered Startup Due Diligence")
st.markdown(
    "Enter a startup name and any supporting context to prepare a "
    "structured, source-backed diligence run."
)

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
    submitted = st.form_submit_button("Prepare analysis", use_container_width=True)

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
