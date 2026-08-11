"""تحميل style.css الموحّد في صفحات Streamlit."""
from pathlib import Path

import streamlit as st

_CSS_PATH = Path(__file__).resolve().parent.parent / "style.css"


def inject_app_styles() -> None:
    """حقن أنماط المشروع من style.css (لا CSS inline في الصفحات)."""
    if not _CSS_PATH.is_file():
        return
    st.markdown(
        f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )
