"""Streamlit UI: upload file → pipeline → display MCQs → download Excel."""
from __future__ import annotations

import html
import os
import tempfile

import streamlit as st

from edu_question_generator.config import (
    DEEPSEEK_MODEL as DEFAULT_DEEPSEEK_MODEL,
    LLM_INSUFFICIENT_BALANCE,
    LLM_INVALID_MODEL,
    LLM_LIMIT_ERROR,
    LLM_REQUEST_TOO_LARGE,
    PIPELINE_ALL_SEGMENTS_FAILED,
    TARGET_QUESTIONS_TOTAL,
)
from edu_question_generator.excel_export import dataframe_to_excel, questions_to_dataframe
from edu_question_generator.generator import detect_lang
from edu_question_generator.loaders import load_text
from edu_question_generator.pipeline import generate_from_document

# Custom CSS for the Edu tab in app.py
_EDU_STYLES = """
<style>
.edu-qg .main-title { font-size: 2.15rem; font-weight: 800; margin-bottom: 0.25rem; text-align: center; }
.edu-qg .sub-title { color: #64748b; margin-bottom: 1.5rem; text-align: center; font-size: 1.08rem; }
.edu-qg .stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    border: 0;
    padding: 0.85rem 1rem;
    font-weight: 600;
    font-size: 1.05rem;
}
.edu-qg .rtl-block { direction: rtl; text-align: right; }
.edu-qg .questions-header { direction: rtl; text-align: right; }
.edu-qg .model-status {
    direction: rtl;
    text-align: right;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 0.5rem;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.95rem;
    color: #334155;
}
</style>
"""

# Pipeline phases shown to the user during generation
_PIPELINE_PHASES: list[tuple[str, str]] = [
    ("extract", "استخراج النص من الملف"),
    ("chunk", "تقسيم المستند"),
    ("generate", "توليد الأسئلة"),
    ("merge", "دمج النتائج"),
    ("validate", "التحقق من الأسئلة"),
    ("select", "اختيار الأسئلة المناسبة"),
    ("done", "اكتمل التوليد"),
]


def _read_secret(name: str) -> str | None:
    """Read a secret from st.secrets, then fall back to environment variable."""
    try:
        value = st.secrets[name]
        if value is None:
            return None
        text = str(value).strip()
        return text or None
    except (KeyError, st.errors.StreamlitSecretNotFoundError, AttributeError):
        pass
    env_value = os.getenv(name)
    if env_value is None:
        return None
    text = env_value.strip()
    return text or None


def get_deepseek_api_key() -> str | None:
    """Return DeepSeek API key for generation."""
    return _read_secret("DEEPSEEK_API_KEY")


def get_deepseek_model() -> str:
    """Return model id from secrets.toml, then env, then config default."""
    return _read_secret("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL


def _init_edu_state() -> None:
    """Initialize session_state keys for the Edu tab."""
    defaults = {
        "edu_questions_df": None,
        "edu_last_filename": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _stage_phase_key(stage: str) -> str:
    """Map internal pipeline stage to display phase (extract/chunk/generate/…)."""
    if stage == "extract_done":
        return "extract"
    if stage == "chunking":
        return "chunk"
    if stage in {"segment_llm_start", "segment_llm_done", "segment_skip"}:
        return "generate"
    if stage in {"merge", "merge_done"}:
        return "merge"
    if stage == "validate":
        return "validate"
    if stage == "cap":
        return "select"
    if stage == "done":
        return "done"
    return "generate"


def pipeline_status_headline(stage: str, data: dict) -> str:
    """Status bar headline shown during generation."""
    del data
    labels = dict(_PIPELINE_PHASES)
    phase = _stage_phase_key(stage)
    label = labels.get(phase, "جاري التوليد")
    if phase == "done":
        return label
    return f"{label}…"


def _render_phase_checklist(completed: set[str], current: str) -> str:
    """HTML checklist for pipeline phases (✓ / ▸ / ○)."""
    rows: list[str] = []
    for key, label in _PIPELINE_PHASES:
        if key in completed:
            rows.append(f"✓ {html.escape(label)}")
        elif key == current:
            rows.append(f"▸ {html.escape(label)}…")
        else:
            rows.append(f"○ {html.escape(label)}")
    return (
        "<div class='rtl-block' style='font-size:0.95rem;line-height:1.9;color:#334155'>"
        + "<br>".join(rows)
        + "</div>"
    )


def make_pipeline_progress_ui():
    """Create pipeline progress callback and progress bar for st.status.

    Returns:
        tuple: (callback, progress_bar) — pass callback to generate_from_document.
    """
    completed_phases: set[str] = set()
    current_phase = "extract"
    log_box = st.empty()
    progress_bar = st.progress(0.0)
    order = [p[0] for p in _PIPELINE_PHASES]

    log_box.markdown(_render_phase_checklist(completed_phases, current_phase), unsafe_allow_html=True)

    def callback(stage: str, data: dict) -> None:
        """Update phase checklist (✓/▸/○) and progress bar on each pipeline event."""
        nonlocal current_phase
        phase = _stage_phase_key(stage)
        if phase in order:
            idx = order.index(phase)
            for earlier in order[:idx]:
                completed_phases.add(earlier)
            current_phase = phase
        log_box.markdown(
            _render_phase_checklist(completed_phases, current_phase),
            unsafe_allow_html=True,
        )
        if stage == "chunking":
            progress_bar.progress(0.12)
        elif stage in {"segment_llm_start", "segment_llm_done", "segment_skip"}:
            index = int(data.get("index") or 1)
            total = max(1, int(data.get("total") or 1))
            progress_bar.progress(0.12 + 0.58 * index / total)
        elif stage in {"merge", "merge_done"}:
            progress_bar.progress(0.78)
        elif stage == "validate":
            progress_bar.progress(0.88)
        elif stage == "cap":
            progress_bar.progress(0.94)
        elif stage == "done":
            completed_phases.update(order)
            current_phase = "done"
            log_box.markdown(
                _render_phase_checklist(completed_phases, "done"),
                unsafe_allow_html=True,
            )
            progress_bar.progress(1.0)

    return callback, progress_bar


def _render_api_key_status() -> None:
    """Show warning when DEEPSEEK_API_KEY is missing."""
    if get_deepseek_api_key():
        return
    st.markdown(
        '<div class="model-status">⚠️ خدمة التوليد غير مهيّأة. '
        "ضع <code>DEEPSEEK_API_KEY</code> في <code>.streamlit/secrets.toml</code> "
        "(وليس config.toml) ثم أعد تشغيل Streamlit.</div>",
        unsafe_allow_html=True,
    )


def _rtl_markdown(content: str) -> None:
    """Render markdown inside an RTL-aligned container."""
    st.markdown(
        f'<div class="rtl-block">{content}</div>',
        unsafe_allow_html=True,
    )


def _render_questions(df) -> None:
    """Display MCQ questions as bordered cards."""
    for _, row in df.iterrows():
        with st.container(border=True):
            kind = row.get("Question Kind", "")
            kind_line = f" · {html.escape(str(kind))}" if kind and str(kind).strip() else ""
            _rtl_markdown(f"<h3>س{row['#']} — اختيار من متعدد{kind_line}</h3>")
            _rtl_markdown(f"<p><strong>{html.escape(str(row['Question']))}</strong></p>")

            options_html = ""
            for letter in ("A", "B", "C", "D"):
                option = row.get(f"Option {letter}", "")
                if option:
                    options_html += (
                        f"<p><strong>{letter}.</strong> {html.escape(str(option))}</p>"
                    )
            _rtl_markdown(options_html)

            _rtl_markdown(
                f"<p>✅ <strong>الإجابة:</strong> {html.escape(str(row['Answer']))}</p>"
            )

            solution = row.get("Solution", row.get("Explanation", ""))
            if solution and str(solution).strip():
                with st.expander("💡 الحل"):
                    _rtl_markdown(f"<p>{html.escape(str(solution))}</p>")


def render_edu_app() -> None:
    """Render the Edu Question Generator UI inside the main app."""
    _init_edu_state()
    st.markdown(_EDU_STYLES, unsafe_allow_html=True)
    st.markdown('<div class="edu-qg">', unsafe_allow_html=True)

    st.markdown(
        '<div class="main-title">Edu Question Generator</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align:center;color:#64748b;font-size:1.08rem;margin-bottom:1.5rem;">'
        "ارفع ملف المحاضرة (PDF / DOCX / TXT) واضغط توليد الأسئلة"
        "</p>",
        unsafe_allow_html=True,
    )

    _render_api_key_status()

    uploaded = st.file_uploader(
        "PDF / DOCX / TXT",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
        key="edu_file_uploader",
    )

    if st.button("توليد الأسئلة", type="primary", use_container_width=True, key="edu_generate"):
        api_key = get_deepseek_api_key()
        if not api_key:
            st.error("خدمة التوليد غير مهيّأة. أضف DEEPSEEK_API_KEY.")
            st.stop()

        if not uploaded:
            st.error("يرجى رفع ملف PDF أو DOCX أو TXT.")
            st.stop()

        with st.status("جاري توليد الأسئلة…", expanded=True) as run_status:
            progress_cb, _progress_bar = make_pipeline_progress_ui()
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name

            def on_pipeline_progress(stage: str, data: dict) -> None:
                """Wire progress callback to st.status label updates."""
                progress_cb(stage, data)
                run_status.update(label=pipeline_status_headline(stage, data))

            try:
                text = load_text(tmp_path).strip()
                if len(text) < 100:
                    run_status.update(label="فشل — نص قصير", state="error")
                    st.error("الملف قصير جداً أو لم يُستخرج منه نص.")
                    st.stop()

                lang = detect_lang(text)
                on_pipeline_progress(
                    "extract_done",
                    {"text_chars": len(text), "lang": "العربية" if lang == "ar" else "English"},
                )

                try:
                    payload, run_meta = generate_from_document(
                        text=text,
                        lang=lang,
                        model=get_deepseek_model(),
                        api_key=api_key,
                        progress_callback=on_pipeline_progress,
                    )
                except RuntimeError as exc:
                    run_status.update(label="فشل التوليد", state="error")
                    message = str(exc)
                    if message == LLM_INSUFFICIENT_BALANCE:
                        st.error(
                            "تعذّر إكمال التوليد (انتهى الرصيد أو الحد المتاح). "
                            "حاول لاحقاً."
                        )
                    elif message == PIPELINE_ALL_SEGMENTS_FAILED:
                        st.error(
                            "تعذّر توليد أسئلة من جميع أجزاء المستند. "
                            "جرّب ملفاً أصغر أو حاول لاحقاً."
                        )
                    elif message == LLM_INVALID_MODEL:
                        st.error(
                            "اسم النموذج غير مدعوم. "
                            "ضع `DEEPSEEK_MODEL` صحيحاً في `.streamlit/secrets.toml` "
                            "(مثل `deepseek-chat` أو `deepseek-reasoner`) "
                            "ثم أعد تشغيل Streamlit."
                        )
                    elif message == LLM_LIMIT_ERROR:
                        st.error("تم استنفاد الحد المتاح. حاول لاحقاً.")
                    elif message == LLM_REQUEST_TOO_LARGE:
                        st.error(
                            "تعذّر معالجة بعض أجزاء المستند. "
                            "جرّب مرة أخرى — النظام يقسّم المستند تلقائياً."
                        )
                    else:
                        st.error("تعذّر توليد الأسئلة. حاول لاحقاً.")
                    st.stop()

                df = questions_to_dataframe(payload)
                st.session_state["edu_questions_df"] = df
                st.session_state["edu_last_filename"] = os.path.splitext(uploaded.name)[0]
                st.session_state["edu_run_meta"] = run_meta
                run_status.update(
                    label=pipeline_status_headline("done", {"mcq_final": len(df)}),
                    state="complete",
                )
            finally:
                os.unlink(tmp_path)

        count = (
            len(st.session_state["edu_questions_df"])
            if st.session_state["edu_questions_df"] is not None
            else 0
        )
        meta = st.session_state.get("edu_run_meta") or {}
        if meta:
            skipped = meta.get("segments_skipped", 0)
            skipped_line = f" — تخطّي {skipped} جزء." if skipped else ""
            st.caption(
                f"النص: {meta.get('text_chars', 0):,} حرف — "
                f"{meta.get('segments_used', 0)} جزء — "
                f"هدف {meta.get('target_questions', TARGET_QUESTIONS_TOTAL)} سؤال "
                f"(ظهر {count}).{skipped_line}"
            )
        if count == 0:
            st.warning("لم يُولَّد أي سؤال صالح من هذا المستند.")
        else:
            st.success(f"تم توليد {count} سؤالاً صالحاً")

    df = st.session_state.get("edu_questions_df")
    if df is not None and not df.empty:
        st.markdown('<h2 class="questions-header">الأسئلة</h2>', unsafe_allow_html=True)
        _render_questions(df)

        excel_bytes = dataframe_to_excel(df)
        base_name = st.session_state.get("edu_last_filename") or "questions"
        st.download_button(
            label="📥 تحميل Excel",
            data=excel_bytes,
            file_name=f"{base_name}_questions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="edu_download_excel",
        )

    st.markdown("</div>", unsafe_allow_html=True)
