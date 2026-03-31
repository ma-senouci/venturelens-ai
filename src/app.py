import streamlit as st

from config import ConfigError, get_settings

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
    "Submit a startup name to receive a structured, source-backed "
    "due diligence memo with an independent review and confidence scoring."
)
st.info("🚧 Application scaffold initialized. Features coming soon.")
