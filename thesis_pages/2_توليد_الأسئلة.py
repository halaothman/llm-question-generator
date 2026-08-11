"""صفحة توليد الأسئلة: Vanilla و RAG، عرض الأسئلة وحفظ JSON عند طلب المستخدم."""
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

try:
    from src.arabic_text import clean_ar
    from src.config import get_index_paths, get_rag_version, set_rag_version
    from src.embeddings import get_model_name
    from src.generator import (
        build_prompt_rag,
        build_prompt_vanilla,
        combined_rag_source_text,
        call_llama,
        detect_lang,
        generate_questions_with_retry,
        safe_json,
    )
    from src.loaders import load_text
    from src.rag import retrieve
    from src.question_counts import count_llm_questions
    from src.storage import save_group, save_questions_separate_file
    from src.ui_styles import inject_app_styles
except ImportError as e:
    st.error(f"خطأ في استيراد الوحدات: {e}")
    st.stop()

st.set_page_config(page_title="توليد الأسئلة", layout="wide")

inject_app_styles()

UPLOADS = "uploads"
os.makedirs(UPLOADS, exist_ok=True)

TEMPERATURE = 0.7
MAX_RETRIES = 3

RAG_SETTINGS = {
    "baseline": {"label": "Baseline (الأصلية)", "threshold": 0.82, "top_k": 5},
    "improved": {"label": "Improved (المحسّنة)", "threshold": 0.65, "top_k": 10},
}

#___________________________________________________________

def rag_settings_for_version(version: str) -> dict:
    """إعدادات RAG (عتبة، top_k، تسمية) حسب النسخة baseline أو improved."""
    return RAG_SETTINGS.get(version, RAG_SETTINGS["baseline"])
#_____________________________________________________________


def _rag_chunk_index(source: dict) -> Optional[int]:
    """استخراج رقم المقطع من metadata مصدر RAG."""
    chunk = (source.get("metadata") or {}).get("chunk_index")
    if chunk is None:
        return None
    try:
        return int(chunk)
    except (TypeError, ValueError):
        return None

#______________________________________________________________

def format_rag_source_line(source: dict, version: str, *, rank: Optional[int] = None) -> str:
    """تنسيق سطر عرض مصدر RAG: رقم المقطع، اسم الملف، ودرجة التشابه."""
    name = source.get("filename", "—")
    if name and ("/" in name or "\\" in name):
        name = os.path.basename(name)
    chunk = _rag_chunk_index(source)
    chunk_label = f"مقطع {chunk}" if chunk is not None else "مقطع —"
    prefix = f"{rank}. " if rank is not None else "- "
    if version == "improved" and "rerank_score" in source:
        faiss = source.get("original_score", source.get("score", 0))
        rerank = source.get("rerank_score", source.get("score", 0))
        return (
            f"{prefix}**{chunk_label}** · `{name}` "
            f"(تشابه FAISS: {faiss:.2f} · إعادة ترتيب: {rerank:.2f})"
        )
    score = source.get("score", 0)
    return f"{prefix}**{chunk_label}** · `{name}` (تشابه FAISS: {score:.2f})"

#______________________________________________________________
def display_retrieved_sources(sources: list[dict], version: str) -> None:
    """عرض قائمة المقاطع المسترجعة مع رقم المقطع واسم الملف ودرجة التشابه."""
    for rank, source in enumerate(sources, 1):
        st.markdown(format_rag_source_line(source, version, rank=rank))

#______________________________________________________________
def ollama_model_names() -> List[str]:
    """قائمة أسماء النماذج المثبتة في Ollama؛ قائمة فارغة عند فشل الاتصال."""
    try:
        import ollama

        listed = ollama.list()
    except Exception:
        return []
    names: List[str] = []
    if listed and hasattr(listed, "models") and hasattr(listed.models, "__iter__"):
        for model in listed.models:
            if hasattr(model, "model"):
                names.append(model.model)
    return names

#______________________________________________________________
def display_questions_with_answers(questions: dict) -> None:
    """عرض أسئلة MCQ وصح/خطأ مع الخيارات والإجابات في الواجهة."""
    if not questions or not isinstance(questions, dict):
        st.info("لا توجد أسئلة للعرض.")
        return

    mcq_list = questions.get("mcq") or []
    if mcq_list:
        st.markdown("#### أسئلة الاختيار من متعدد")
        for i, mcq in enumerate(mcq_list, 1):
            if not isinstance(mcq, dict):
                continue
            q_text = mcq.get("q") or mcq.get("question") or ""
            if not q_text:
                continue
            st.markdown(f"**{i}.** {q_text}")
            for opt in mcq.get("options") or []:
                st.markdown(f"- {opt}")
            st.markdown(f"**الإجابة الصحيحة:** {mcq.get('answer', '—')}")
            st.divider()

    tf_list = questions.get("tf") or []
    if tf_list:
        st.markdown("#### أسئلة صح / خطأ")
        for i, tf in enumerate(tf_list, 1):
            if not isinstance(tf, dict):
                continue
            q_text = tf.get("q") or tf.get("question") or ""
            if not q_text:
                continue
            ans = tf.get("answer")
            if isinstance(ans, bool):
                ans_label = "صح" if ans else "خطأ"
            else:
                ans_label = str(ans)
            st.markdown(f"**{i}.** {q_text}")
            st.markdown(f"**الإجابة:** {ans_label}")
            st.divider()

#______________________________________________________________
def parse_llm_questions(
    prompt: str,
    raw_text: str,
    model_name: str,
    lang: str,
    retrieved=None,
) -> Optional[dict]:
    """استدعاء LLM وتحليل JSON؛ مع إعادة محاولة عند فشل التنسيق.

    Args:
        prompt: نص البرومبت المرسل للنموذج.
        raw_text: النص المصدر (للمقاييس والإصلاح).
        model_name: اسم نموذج Ollama.
        lang: لغة المستند (ar/en).
        retrieved: مقاطع RAG المسترجعة (None في Vanilla).

    Returns:
        dict يحتوي mcq و tf، أو None عند الفشل.
    """
    response = call_llama(prompt, model_name=model_name, temperature=TEMPERATURE)
    out = safe_json(response, raw_text, model_name, lang, retrieved)
    if out is None:
        out = generate_questions_with_retry(
            prompt,
            max_retries=MAX_RETRIES,
            source_text=raw_text,
            model_name=model_name,
            lang=lang,
            retrieved=retrieved,
        )
    if out is None or not isinstance(out, dict) or "mcq" not in out or "tf" not in out:
        return None
    if not (out.get("mcq") or out.get("tf")):
        return None
    return out

#______________________________________________________________
def save_output(out: dict, model_name: str, method: str, upload_name: str, lang: str) -> None:
    """حفظ الأسئلة في ملف JSON منفصل؛ مع fallback إلى outputs/questions.json."""
    try:
        filename, num_questions = save_questions_separate_file(
            out, model_name, method, upload_name, lang,
        )
        st.success(f"تم حفظ {num_questions} سؤال في `{filename}`")
    except Exception as e:
        st.error(f"خطأ في حفظ الملف: {e}")
        group = "A" if method == "vanilla" else "B"
        save_group(
            group,
            {
                "lang": lang,
                **out,
                "source_file": upload_name,
                "generation_time": out.get("generation_time", 0.0),
                "method": method,
            },
        )
        st.warning("تم الحفظ الاحتياطي في outputs/questions.json")

#______________________________________________________________
def run_vanilla(raw_text: str, upload_name: str, model_name: str, lang: str) -> None:
    """توليد أسئلة Vanilla من النص المرفوع فقط (بدون RAG)."""
    with st.spinner("جاري توليد الأسئلة (Vanilla)..."):
        try:
            prompt = build_prompt_vanilla(raw_text)
            start = time.time()
            out = parse_llm_questions(prompt, raw_text, model_name, lang, retrieved=None)
            if out is None:
                st.error(
                    "فشل توليد الأسئلة أو تنسيق JSON. "
                    "تأكد أن Ollama يعمل والنموذج المختار متاح، ثم أعد المحاولة."
                )
                return
            out["source_text"] = raw_text
            out["generation_time"] = time.time() - start
            st.caption(
                f"عدد الأسئلة: {count_llm_questions(out)} — "
                f"الوقت: {out['generation_time']:.1f} ث"
            )
            st.session_state.vanilla_questions = out
            st.session_state.vanilla_generated = True
            st.session_state.vanilla_saved = False
            st.session_state.vanilla_context = {
                "upload_name": upload_name,
                "lang": lang,
                "model_name": model_name,
            }
        except Exception as e:
            st.error(f"خطأ في توليد الأسئلة: {e}")


def run_rag(
    raw_text: str,
    upload_name: str,
    model_name: str,
    lang: str,
    idx_path: str,
    meta_path: str,
) -> None:
    """توليد أسئلة RAG: استرجاع من الفهرس ثم توليد من النص + المقاطع."""
    with st.spinner("جاري الاسترجاع وتوليد الأسئلة (RAG)..."):
        try:
            version = get_rag_version()
            cfg = rag_settings_for_version(version)

            if version == "baseline":
                retrieved = retrieve(
                    idx_path,
                    meta_path,
                    raw_text,
                    top_k=cfg["top_k"],
                    thr=cfg["threshold"],
                )
            else:
                retrieved = retrieve(idx_path, meta_path, raw_text)

            if not retrieved:
                st.error("لم يُعثر على مقاطع مشابهة بدرجة كافية في الفهرس.")
                st.info(
                    f"عتبة النسخة الحالية ({cfg['label']}): {cfg['threshold']}. "
                    "أضف مصادر في «فهرسة قاعدة المعرفة» أو جرّب Vanilla."
                )
                return

            st.markdown("**المقاطع المسترجعة:**")
            display_retrieved_sources(retrieved, version)

            rag_source_text = combined_rag_source_text(raw_text, retrieved)
            prompt = build_prompt_rag(raw_text, retrieved)

            start = time.time()
            out = parse_llm_questions(
                prompt, rag_source_text, model_name, lang, retrieved=retrieved
            )
            if out is None:
                st.error(
                    "فشل توليد الأسئلة أو تنسيق JSON. "
                    "تأكد أن Ollama يعمل والنموذج المختار متاح، ثم أعد المحاولة."
                )
                return

            out["sources"] = retrieved
            out["source_text"] = rag_source_text
            out["upload_file"] = upload_name
            out["generation_time"] = time.time() - start

            st.caption(
                f"مقاطع RAG: {len(retrieved)} — عدد الأسئلة: {count_llm_questions(out)} — "
                f"الوقت: {out['generation_time']:.1f} ث"
            )
            if version == "baseline":
                st.caption(
                    f"Baseline: FAISS top-{cfg['top_k']} → عتبة {cfg['threshold']} — "
                    "التوليد من الملف المرفوع + المقاطع المسترجعة (بدون إعادة ترتيب)."
                )
            else:
                st.caption(
                    f"Improved: FAISS → عتبة {cfg['threshold']} → top-{cfg['top_k']} → "
                    f"re-rank top-5 — التوليد من الملف المرفوع + المقاطع المسترجعة."
                )
            st.session_state.rag_questions = out
            st.session_state.rag_generated = True
            st.session_state.rag_saved = False
            st.session_state.rag_context = {
                "upload_name": upload_name,
                "lang": lang,
                "model_name": model_name,
            }
        except Exception as e:
            st.error(f"خطأ في توليد الأسئلة: {e}")


# --- واجهة الصفحة ---
st.title("توليد الأسئلة (Vanilla & RAG)")

col1, col2 = st.columns(2)
with col1:
    if st.button("LLaMA 3.2:3B", use_container_width=True, type="primary"):
        st.session_state.selected_model = "llama3.2:3b"
with col2:
    if st.button("Qwen 2.5:7B", use_container_width=True, type="primary"):
        st.session_state.selected_model = "qwen2.5:7b"

if st.session_state.get("selected_model"):
    st.caption(f"النموذج: **{st.session_state.selected_model}**")
else:
    st.warning("اختر نموذجاً أولاً")

st.markdown("---")

idx_path, meta_path = get_index_paths()
current_version = get_rag_version()

with st.sidebar:
    st.header("إعدادات النظام")

    st.subheader("Ollama")
    installed = ollama_model_names()
    llama_ok = any("llama3.2:3b" in name for name in installed)
    qwen_ok = any("qwen2.5:7b" in name for name in installed)
    if llama_ok and qwen_ok:
        st.success("LLaMA و Qwen متصلان")
    elif installed:
        if llama_ok:
            st.warning("LLaMA متصل — Qwen غير متاح")
        elif qwen_ok:
            st.warning("Qwen متصل — LLaMA غير متاح")
        else:
            st.warning("Ollama متصل — LLaMA و Qwen غير متاحين")
    else:
        st.warning("تعذّر الاتصال بـ Ollama")

    st.markdown("---")
    st.subheader("نسخة RAG")
    selected_version = st.radio(
        "النسخة",
        options=["baseline", "improved"],
        index=0 if current_version == "baseline" else 1,
        format_func=lambda v: rag_settings_for_version(v)["label"],
        key="rag_version_selector",
    )
    if selected_version != current_version:
        set_rag_version(selected_version)
        st.rerun()

    idx_path, meta_path = get_index_paths()

    st.caption(rag_settings_for_version(get_rag_version())["label"])
    st.caption(f"التضمين: `{get_model_name()}`")
    cfg = rag_settings_for_version(get_rag_version())
    if get_rag_version() == "baseline":
        st.caption(f"FAISS top-{cfg['top_k']} · عتبة {cfg['threshold']} · بدون re-ranking")
    else:
        st.caption(f"عتبة {cfg['threshold']} · top-{cfg['top_k']} فوق العتبة · re-rank → 5")

    if st.session_state.get("selected_model"):
        st.metric("النموذج المختار", st.session_state.selected_model)

st.subheader("رفع الملف")
if "selected_model" not in st.session_state:
    st.warning("اختر نموذجاً من الأعلى.")

up = st.file_uploader("ارفع PDF / DOCX / TXT / MD", type=["pdf", "docx", "doc", "txt", "md"])
if not up:
    st.info("ارفع ملفاً لإظهار أزرار التوليد.")
    st.stop()

path = os.path.join(UPLOADS, up.name)
with open(path, "wb") as f:
    f.write(up.getbuffer())

try:
    raw_text = load_text(path)
    if not raw_text.strip():
        st.error("الملف فارغ أو لا يحتوي على نص")
        st.stop()
    lang = detect_lang(raw_text)
    model_name = st.session_state.get("selected_model")
except Exception as e:
    st.error(f"خطأ في قراءة الملف: {e}")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    st.metric("الملف", up.name)
with c2:
    st.metric("اللغة", "العربية" if lang == "ar" else "English")

st.subheader("اختر طريقة التوليد")
col_v, col_r = st.columns(2)

with col_v:
    st.markdown("#### Vanilla")
    st.caption("من النص المرفوع فقط")
    if st.button("توليد Vanilla", type="primary", use_container_width=True):
        if not model_name:
            st.error("اختر نموذجاً أولاً.")
        else:
            run_vanilla(raw_text, up.name, model_name, lang)

with col_r:
    st.markdown("#### RAG")
    st.caption("مع استرجاع من المراجع المفهرسة")
    if not os.path.exists(idx_path):
        st.warning("فهرس قاعدة المعرفة غير موجود — استخدم «فهرسة قاعدة المعرفة».")
    elif st.button("توليد RAG", type="primary", use_container_width=True):
        if not model_name:
            st.error("اختر نموذجاً أولاً.")
        else:
            run_rag(raw_text, up.name, model_name, lang, idx_path, meta_path)

if st.session_state.get("vanilla_generated"):
    st.subheader("الأسئلة (Vanilla)")
    display_questions_with_answers(st.session_state.get("vanilla_questions"))
    ctx = st.session_state.get("vanilla_context") or {}
    if st.button("حفظ أسئلة Vanilla", type="primary", key="save_vanilla"):
        out = st.session_state.get("vanilla_questions")
        if not out or not ctx.get("model_name"):
            st.error("لا توجد أسئلة للحفظ.")
        else:
            with st.spinner("جاري حساب المقاييس وحفظ الملف..."):
                save_output(
                    out,
                    ctx["model_name"],
                    "vanilla",
                    ctx["upload_name"],
                    ctx["lang"],
                )
            st.session_state.vanilla_saved = True
    if st.session_state.get("vanilla_saved"):
        st.caption("✓ تم الحفظ على القرص.")

if st.session_state.get("rag_generated"):
    st.subheader("الأسئلة (RAG)")
    rag_out = st.session_state.get("rag_questions") or {}
    sources = rag_out.get("sources") or []
    if sources:
        version = get_rag_version()
        st.markdown("**مصادر RAG:**")
        display_retrieved_sources(sources[:8], version)
        if version == "improved" and any("rerank_score" in s for s in sources):
            st.caption(
                "«إعادة ترتيب» درجة صلة من Cross-Encoder وليست نسبة تشابه؛ "
                "المهم ترتيب المقاطع لا القيمة المطلقة."
            )
    display_questions_with_answers(rag_out)
    ctx = st.session_state.get("rag_context") or {}
    if st.button("حفظ أسئلة RAG", type="primary", key="save_rag"):
        if not rag_out or not ctx.get("model_name"):
            st.error("لا توجد أسئلة للحفظ.")
        else:
            with st.spinner("جاري حساب المقاييس وحفظ الملف..."):
                save_output(
                    rag_out,
                    ctx["model_name"],
                    "rag",
                    ctx["upload_name"],
                    ctx["lang"],
                )
            st.session_state.rag_saved = True
    if st.session_state.get("rag_saved"):
        st.caption("✓ تم الحفظ على القرص.")
