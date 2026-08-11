"""
نسخة Baseline: نموذج التضمينات الأصلي (قبل التحسين)
- النموذج: intfloat/e5-large-v2 (إنجليزي)
- الإعدادات: الأصلية
"""
from sentence_transformers import SentenceTransformer
import numpy as np

# متغير عام لحفظ النموذج
_model = None

# نموذج التضمين الأصلي (Baseline)
BASELINE_MODEL_NAME = "intfloat/e5-large-v2"

def get_model():
    """
    الحصول على نموذج التضمين (lazy loading) - Baseline
    
    Returns:
        SentenceTransformer: نموذج التضمين
    """
    global _model
    if _model is None:
        print(f"[BASELINE] جاري تحميل النموذج الإنجليزي: {BASELINE_MODEL_NAME}")
        _model = SentenceTransformer(BASELINE_MODEL_NAME)
        print("[BASELINE] تم تحميل النموذج بنجاح")
    return _model

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    تحويل النصوص إلى تضمينات - Baseline
    Baseline / e5-large-v2: لا تمييز بين query و passage في هذه الدالة.
    
    Args:
        texts: قائمة النصوص المراد تحويلها
    
    Returns:
        np.ndarray: مصفوفة التضمينات
    """
    if not texts:
        return np.array([])
    
    model = get_model()
    
    # e5-large-v2 (Baseline): بادئة passage: للمقاطع (نفس التنسيق للاستعلام هنا)
    formatted_texts = [f"passage: {text}" for text in texts]
    
    # تحويل النصوص إلى تضمينات
    embeddings = model.encode(
        formatted_texts, 
        normalize_embeddings=True, 
        show_progress_bar=False
    )
    
    return np.array(embeddings)

def get_model_name():
    """الحصول على اسم النموذج"""
    return BASELINE_MODEL_NAME







