"""
نسخة Improved: نموذج التضمينات المحسّن (بعد التحسين)
- النموذج: aubmindlab/bert-base-arabertv2 (عربي محسّن)
- الإعدادات: محسّنة
"""
from sentence_transformers import SentenceTransformer
import numpy as np

# متغير عام لحفظ النموذج
_model = None

# نموذج التضمين المحسّن (Improved)
IMPROVED_MODEL_NAME = "aubmindlab/bert-base-arabertv2"

def get_model():
    """
    الحصول على نموذج التضمين (lazy loading) - Improved
    
    Returns:
        SentenceTransformer: نموذج التضمين
    """
    global _model
    if _model is None:
        print(f"[IMPROVED] جاري تحميل النموذج العربي: {IMPROVED_MODEL_NAME}")
        _model = SentenceTransformer(IMPROVED_MODEL_NAME)
        print("[IMPROVED] تم تحميل النموذج بنجاح")
    return _model

def embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    """
    تحويل النصوص إلى تضمينات - Improved
    (يدعم التمييز بين query و passage)
    
    Args:
        texts: قائمة النصوص المراد تحويلها
        is_query: إذا كان True، فهذا استعلام (للبحث)
                 إذا كان False، فهذا مقطع (للفهرسة)
    
    Returns:
        np.ndarray: مصفوفة التضمينات
    """
    if not texts:
        return np.array([])
    
    model = get_model()
    
    # النماذج العربية لا تحتاج تنسيق خاص (مثل "query:" أو "passage:")
    # يمكن استخدام النصوص مباشرة
    formatted_texts = texts
    
    # تحويل النصوص إلى تضمينات
    embeddings = model.encode(
        formatted_texts, 
        normalize_embeddings=True, 
        show_progress_bar=False
    )
    
    return np.array(embeddings)

def get_model_name():
    """الحصول على اسم النموذج"""
    return IMPROVED_MODEL_NAME







