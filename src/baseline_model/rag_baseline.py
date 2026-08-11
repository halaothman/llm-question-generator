"""
نسخة Baseline: RAG الأصلي (قبل التحسين)
- TOP_K: 5 (مرشّحو FAISS قبل فلترة العتبة)
- SIMILARITY_THRESHOLD: 0.82
- نموذج التضمين: e5-large-v2
- بدون إعادة ترتيب
"""
import os

import numpy as np

from ..arabic_text import clean_ar, stem_ar
from ..faiss_store import load_meta, resolve_chunk_index, search
from .embeddings_baseline import embed_texts

# إعدادات Baseline
BASELINE_TOP_K = 5
BASELINE_THRESHOLD = 0.82


def _prepare_query(query: str) -> str:
    """تنظيف وتجذيع الاستعلام لمطابقة متجهات الفهرس (embed_text)."""
    return stem_ar(clean_ar(query))


def retrieve(index_path: str, meta_path: str, query: str, top_k: int = BASELINE_TOP_K, thr: float = BASELINE_THRESHOLD):
    """
    استرجاع المقاطع المشابهة من الفهرس - Baseline (النسخة الأصلية)
    
    Args:
        index_path: مسار ملف الفهرس
        meta_path: مسار ملف الميتاداتا
        query: نص الاستعلام
        top_k: عدد مرشّحي FAISS (افتراضي: 5 — Baseline)
        thr: عتبة التشابه (افتراضي: 0.82 — Baseline)
    
    Returns:
        list: قائمة المقاطع المسترجعة
    """
    prepared = _prepare_query(query)
    if not prepared.strip():
        return []

    # Baseline (e5-large-v2): لا تمييز query/passage — الاستعلام يُضمَّن كـ passage أيضاً
    qv = embed_texts([prepared])
    
    # البحث في الفهرس
    scores, ids = search(index_path, qv, top_k)
    
    # تحميل الميتاداتا
    meta = load_meta(meta_path)
    
    # معالجة النتائج
    results = []
    
    # التحقق من أن النتائج ليست فارغة
    if len(scores) == 0 or len(ids) == 0:
        return results
    
    # التأكد من أن scores و ids هما arrays
    if not hasattr(scores, '__iter__'):
        scores = [scores]
    if not hasattr(ids, '__iter__'):
        ids = [ids]
    
    for score, idx in zip(scores, ids):
        if idx == -1:
            continue
        
        # تحويل score إلى float بأمان
        try:
            score = float(score)
        except (ValueError, TypeError):
            continue
        
        if score >= thr:
            metadata = meta.get(int(idx), {})
            meta_inner = metadata.get("metadata", {})
            chunk_index = resolve_chunk_index(meta, int(idx), meta_inner)
            filename = meta_inner.get("filename") or ""
            source = meta_inner.get("source") or ""
            if not filename and source:
                filename = os.path.basename(source)
            results.append({
                "text": metadata.get("text", ""),
                "filename": filename or source,
                "score": score,
                "id": int(idx),
                "metadata": {**meta_inner, "chunk_index": chunk_index},
            })
    
    return results







