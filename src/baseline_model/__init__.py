"""
Baseline Model: النسخة الأصلية قبل التحسين
- نموذج التضمين: intfloat/e5-large-v2
- TOP_K: 5
- SIMILARITY_THRESHOLD: 0.82
"""
from .embeddings_baseline import get_model, embed_texts, get_model_name
from .rag_baseline import retrieve

__all__ = ['get_model', 'embed_texts', 'get_model_name', 'retrieve']







