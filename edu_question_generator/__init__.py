"""Edu Question Generator — MCQ generation via DeepSeek API (separate from the thesis Ollama project)."""

from .ui import render_edu_app

# Public entry point consumed by app.py (Edu tab)
__all__ = ["render_edu_app"]
