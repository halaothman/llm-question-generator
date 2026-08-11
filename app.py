"""نقطة دخول Streamlit: Edu Question Generator + صفحات مشروع الرسالة (pages/)."""
import logging

import streamlit as st
import streamlit_path  # noqa: F401 — جذر المشروع على sys.path

logging.getLogger().setLevel(logging.ERROR)

st.set_page_config(
    page_title="Edu Question Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.ui_styles import inject_app_styles

inject_app_styles()

from edu_question_generator.ui import render_edu_app

render_edu_app()
