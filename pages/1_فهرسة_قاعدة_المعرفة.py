"""صفحة فهرسة قاعدة المعرفة: تقسيم النص وبناء فهرس FAISS من data-sources."""
import os
import sys
import time
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(page_title="فهرسة قاعدة المعرفة", layout="wide")

try:
    from src.loaders import load_text
    from src.chunking import chunk_text
    from src.arabic_text import clean_ar, stem_ar
    from src.faiss_store import build_index, clear_index
except ImportError as e:
    st.error(f"خطأ في استيراد الوحدات: {e}")
    st.stop()

from src.ui_styles import inject_app_styles

inject_app_styles()

st.title("فهرسة قاعدة المعرفة (RAG)")

DATA_DIR = "data-sources"
os.makedirs("indexes", exist_ok=True)

from src.config import get_index_paths, get_rag_version, set_rag_version
from src.embeddings import get_model_name


def _records_for_file(path: str, f: str) -> list[dict]:
    """تحميل ملف مصدر وتقسيمه إلى سجلات جاهزة للفهرسة."""
    text = load_text(path)
    if not text.strip():
        return []

    text = clean_ar(text)
    chunks = chunk_text(text, 500, 100)
    is_improved = get_rag_version() == "improved"

    records = []
    for chunk_idx, ch in enumerate(chunks):
        if not ch.strip():
            continue
        rid = int(uuid.uuid4().int % 1_000_000_000)
        metadata = {
            "source": path,
            "filename": f,
            "chunk_index": chunk_idx + 1,
            "chunk_size": len(ch),
        }
        if is_improved:
            chunk_start = chunk_idx * (500 - 100)
            chunk_end = min(chunk_start + len(ch), len(text))
            context_before = text[max(0, chunk_start - 100) : chunk_start]
            context_after = text[chunk_end : min(chunk_end + 100, len(text))]
            metadata.update(
                {
                    "chunk_position": {
                        "start": chunk_start,
                        "end": chunk_end,
                        "length": len(ch),
                    },
                    "context": (context_before + " ... " + context_after).strip(),
                    "file_type": os.path.splitext(f)[1].lower(),
                }
            )
        records.append(
            {
                "id": rid,
                "text": ch,
                "embed_text": stem_ar(ch),
                "metadata": metadata,
            }
        )
    return records


def _list_source_files(data_dir: str) -> list[str]:
    files = []
    for root, _, names in os.walk(data_dir):
        for name in names:
            files.append(os.path.join(root, name))
    return files


IDX_PATH, META_PATH = get_index_paths()
current_version = get_rag_version()
embedding_model_name = get_model_name()

st.subheader("نسخة RAG")
version_options = {
    "baseline": " Baseline (الأصلية)",
    "improved": " Improved (المحسّنة)",
}
selected_version = st.radio(
    "اختر النسخة للفهرسة:",
    options=["baseline", "improved"],
    index=0 if current_version == "baseline" else 1,
    format_func=lambda x: version_options[x],
    key="rag_version_indexing",
)

if selected_version != current_version:
    set_rag_version(selected_version)
    st.rerun()

IDX_PATH, META_PATH = get_index_paths()

version_info = {
    "baseline": "Baseline (الأصلية)",
    "improved": "Improved (المحسّنة)",
}
st.caption(f"{version_info[current_version]} — `{embedding_model_name}`")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("عدد ملفات المصادر", len(_list_source_files(DATA_DIR)))
with col2:
    try:
        meta_lines = sum(1 for _ in open(META_PATH, "r", encoding="utf-8"))
    except OSError:
        meta_lines = 0
    st.metric("عدد القطع المفهرسة", meta_lines)
with col3:
    st.metric("حالة الفهرس", "موجود" if os.path.exists(IDX_PATH) else "غير موجود")

st.subheader("عمليات الفهرسة")
st.caption(
    "زر **فهرسة قاعدة المعرفة** يمسح الفهرس الحالي ويعيد بناءه من كل الملفات في `data-sources/`."
)

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    run_index = st.button("فهرسة قاعدة المعرفة", type="primary", use_container_width=True)
with btn_col2:
    clear_index_btn = st.button("مسح الفهرس", type="secondary", use_container_width=True)

if clear_index_btn:
    try:
        clear_index(IDX_PATH, META_PATH)
        st.success("تم مسح الفهرس")
        st.rerun()
    except OSError as e:
        st.error(f"خطأ: {e}")

if run_index:
    files_to_index = _list_source_files(DATA_DIR)
    if not files_to_index:
        st.warning("لا توجد ملفات في مجلد data-sources")
    else:
        clear_index(IDX_PATH, META_PATH)
        progress = st.progress(0.0)
        status = st.empty()
        files_ok = 0
        total_chunks = 0
        batch_size = 16
        pending: list[dict] = []
        start = time.time()

        for idx, path in enumerate(files_to_index):
            name = os.path.basename(path)
            status.markdown(f"**{idx + 1}/{len(files_to_index)}** — `{name}`")
            progress.progress((idx + 1) / len(files_to_index))
            try:
                file_records = _records_for_file(path, name)
                if not file_records:
                    st.warning(f"ملف فارغ أو بلا قطع: {name}")
                    continue
                files_ok += 1
                total_chunks += len(file_records)
                pending.extend(file_records)
                while len(pending) >= batch_size:
                    batch = pending[:batch_size]
                    del pending[:batch_size]
                    build_index(IDX_PATH, META_PATH, batch, append=True)
            except Exception as e:
                st.error(f"خطأ في {name}: {e}")

        if pending:
            build_index(IDX_PATH, META_PATH, pending, append=True)

        elapsed = time.time() - start
        if total_chunks:
            st.success(
                f"تمت فهرسة **{total_chunks}** قطعة من **{files_ok}** ملف في **{elapsed:.1f}** ثانية."
            )
        else:
            st.error("لم يتم إنشاء أي قطع للفهرسة")
