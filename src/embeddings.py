"""
واجهة موحدة للتضمينات - تختار تلقائياً بين Baseline و Improved
"""
from .config import get_rag_version

def _get_embeddings_module():
    """الحصول على وحدة التضمينات المناسبة"""
    version = get_rag_version()
    if version == "baseline":
        from .baseline_model import embeddings_baseline as emb_module
        return emb_module
    else:  # improved
        from .improved_model import embeddings_improved as emb_module
        return emb_module

# إعادة تصدير الدوال من الوحدة المناسبة
def get_model():
    """الحصول على نموذج التضمين"""
    module = _get_embeddings_module()
    return module.get_model()

def embed_texts(texts, is_query=False):
    """تحويل النصوص إلى تضمينات"""
    module = _get_embeddings_module()
    # Baseline لا يدعم is_query، Improved يدعمه
    import inspect
    sig = inspect.signature(module.embed_texts)
    if 'is_query' in sig.parameters:
        return module.embed_texts(texts, is_query=is_query)
    else:
        # Baseline - تجاهل is_query
        return module.embed_texts(texts)

def get_model_name():
    """الحصول على اسم النموذج"""
    module = _get_embeddings_module()
    return module.get_model_name()

def get_model_info():
    """الحصول على معلومات النموذج"""
    module = _get_embeddings_module()
    model = get_model()
    return {
        "name": get_model_name(),
        "version": get_rag_version(),
        "loaded": model is not None,
        "model_type": type(model).__name__ if model else None
    }
