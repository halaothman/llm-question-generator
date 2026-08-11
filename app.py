"""Streamlit home: Edu Question Generator. Thesis pages live in pages/ (auto sidebar)."""
import logging

import streamlit as st
import streamlit_path  # noqa: F401 — project root on sys.path

logging.getLogger().setLevel(logging.ERROR)

st.set_page_config(
    page_title="Edu Question Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.ui_styles import inject_app_styles
from edu_question_generator.ui import render_edu_app

inject_app_styles()
render_edu_app()
