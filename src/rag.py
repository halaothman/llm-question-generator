"""
واجهة موحدة لـ RAG - تختار تلقائياً بين Baseline و Improved
"""
from .config import get_rag_version

def _get_rag_module():
    """الحصول على وحدة RAG المناسبة"""
    version = get_rag_version()
    if version == "baseline":
        from .baseline_model import rag_baseline as rag_module
        return rag_module
    else:  # improved
        from .improved_model import rag_improved as rag_module
        return rag_module

def retrieve(index_path: str, meta_path: str, query: str, top_k: int = None, thr: float = None):
    """
    استرجاع المقاطع المشابهة من الفهرس
    يستخدم الإعدادات الافتراضية للنسخة المحددة
    """
    module = _get_rag_module()
    kwargs = {}
    if top_k is not None:
        kwargs["top_k"] = top_k
    if thr is not None:
        kwargs["thr"] = thr
    return module.retrieve(index_path, meta_path, query, **kwargs)
