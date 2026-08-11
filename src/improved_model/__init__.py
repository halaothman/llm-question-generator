"""
Improved Model: النسخة المحسّنة بعد التحسين
- نموذج التضمين: aubmindlab/bert-base-arabertv2
- TOP_K: 10
- SIMILARITY_THRESHOLD: 0.65
- Re-ranking: Cross-Encoder
- Metadata محسّن
"""
from .embeddings_improved import get_model, embed_texts, get_model_name
from .rag_improved import retrieve
from .reranking import rerank, get_reranker

__all__ = ['get_model', 'embed_texts', 'get_model_name', 'retrieve', 'rerank', 'get_reranker']

