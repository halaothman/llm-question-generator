"""
نسخة Improved: RAG المحسّن (بعد التحسين)
- SIMILARITY_THRESHOLD: 0.65
- TOP_K فوق العتبة: 10 (ثم Re-ranking → 5)
- نموذج التضمين: AraBERT
- Re-ranking: Cross-Encoder
"""
import os

import numpy as np

from ..arabic_text import clean_ar, stem_ar
from ..faiss_store import load_meta, resolve_chunk_index, search
from .embeddings_improved import embed_texts
from .reranking import rerank

# إعدادات Improved
IMPROVED_THRESHOLD = 0.65
IMPROVED_FAISS_SEARCH_K = 100  # نطاق FAISS للعثور على ≥ top_k فوق العتبة
IMPROVED_TOP_K = 10  # أقصى عدد فوق العتبة يُمرَّر لإعادة الترتيب
IMPROVED_RERANK_TOP_K = 5  # عدد النتائج النهائية بعد Re-ranking


def _prepare_query(query: str) -> str:
    """تنظيف وتجذيع الاستعلام لمطابقة متجهات الفهرس (embed_text)."""
    return stem_ar(clean_ar(query))


def retrieve(index_path: str, meta_path: str, query: str, top_k: int = IMPROVED_TOP_K, thr: float = IMPROVED_THRESHOLD):
    """
    استرجاع المقاطع المشابهة من الفهرس - Improved (النسخة المحسّنة)
    مع Re-ranking دائماً

    Args:
        index_path: مسار ملف الفهرس
        meta_path: مسار ملف الميتاداتا
        query: نص الاستعلام
        top_k: أقصى عدد فوق العتبة قبل Re-ranking (افتراضي: 10)
        thr: عتبة التشابه (افتراضي: 0.65 - Improved)

    Returns:
        list: أفضل IMPROVED_RERANK_TOP_K مقاطع بعد Re-ranking
    """
    prepared = _prepare_query(query)
    if not prepared.strip():
        return []

    qv = embed_texts([prepared], is_query=True)

    faiss_k = max(IMPROVED_FAISS_SEARCH_K, top_k * 10)
    scores, ids = search(index_path, qv, faiss_k)

    meta = load_meta(meta_path)
    results = []

    # لا نتائج من FAISS (فهرس فارغ أو بحث فاشل)
    if len(scores) == 0 or len(ids) == 0:
        return results

    # FAISS قد يُرجع قيمة واحدة عند k=1 — نُحوّلها لقائمة للتكرار
    if not hasattr(scores, "__iter__"):
        scores = [scores]
    if not hasattr(ids, "__iter__"):
        ids = [ids]

    # المرور على المرشّحين بترتيب تشابه FAISS (من الأعلى للأقل)
    for score, idx in zip(scores, ids):
        # -1 = خانة فارغة في FAISS (أقل من k نتائج في الفهرس)
        if idx == -1:
            continue

        # تحويل الدرجة إلى float (numpy scalar أو نص)
        try:
            score = float(score)
        except (ValueError, TypeError):
            continue

        # قبول المقطع فقط إذا تجاوز عتبة Improved (0.65)
        if score >= thr:
            metadata = meta.get(int(idx), {})
            meta_dict = metadata.get("metadata", {})
            chunk_index = resolve_chunk_index(meta, int(idx), meta_dict)
            source = meta_dict.get("source", "")
            results.append({
                "text": metadata.get("text", ""),  # نص طبيعي للعرض والتوليد
                "filename": os.path.basename(source) if source else meta_dict.get("filename", ""),
                "score": score,  # تشابه FAISS قبل re-rank
                "id": int(idx),
                "metadata": {**meta_dict, "chunk_index": chunk_index},
            })

    results = results[:top_k]
    return rerank(clean_ar(query), results, top_k=IMPROVED_RERANK_TOP_K)
