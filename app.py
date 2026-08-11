"""Streamlit entry: Edu Question Generator + thesis RAG pages (sidebar navigation)."""
import logging
from pathlib import Path

import streamlit as st
import streamlit_path  # noqa: F401 — project root on sys.path

logging.getLogger().setLevel(logging.ERROR)

_ROOT = Path(__file__).resolve().parent
_PAGES = _ROOT / "thesis_pages"


def _edu_main() -> None:
    from edu_question_generator.ui import render_edu_app

    render_edu_app()


st.set_page_config(
    page_title="Edu Question Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.ui_styles import inject_app_styles

inject_app_styles()

pg = st.navigation(
    [
        st.Page(_edu_main, title="Edu Question Generator", icon="📝", default=True),
        st.Page(_PAGES / "1_فهرسة_قاعدة_المعرفة.py", title="فهرسة قاعدة المعرفة", icon="📚"),
        st.Page(_PAGES / "2_توليد_الأسئلة.py", title="توليد الأسئلة", icon="✨"),
        st.Page(_PAGES / "3_المقارنة_والتحليل.py", title="التحليل والمقارنة", icon="📊"),
        st.Page(_PAGES / "5_التقييم_البشري.py", title="التقييم البشري", icon="👥"),
    ]
)
pg.run()
