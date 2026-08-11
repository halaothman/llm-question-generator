"""
ملف الإعدادات المركزي للتبديل بين Baseline و Improved
"""
import os

# تحديد النسخة المستخدمة
# القيم الممكنة: "baseline" أو "improved"
RAG_VERSION = os.getenv("RAG_VERSION", "baseline")  # افتراضي: baseline

def get_rag_version():
    """الحصول على النسخة المحددة"""
    return RAG_VERSION

def set_rag_version(version: str):
    """تعيين النسخة (baseline أو improved)"""
    global RAG_VERSION
    if version in ["baseline", "improved"]:
        RAG_VERSION = version
        os.environ["RAG_VERSION"] = version
    else:
        raise ValueError(f"النسخة يجب أن تكون 'baseline' أو 'improved'، حصلت: {version}")

def get_index_paths():
    """
    الحصول على مسارات الفهرس بناءً على النسخة المحددة
    
    Returns:
        tuple: (index_path, meta_path) - مسار الفهرس ومسار الميتاداتا
    """
    version = get_rag_version()
    base_dir = "indexes"
    
    if version == "baseline":
        index_path = os.path.join(base_dir, "baseline.external.index")
        meta_path = os.path.join(base_dir, "baseline.external_meta.jsonl")
    else:  # improved
        index_path = os.path.join(base_dir, "improved.external.index")
        meta_path = os.path.join(base_dir, "improved.external_meta.jsonl")
    
    # التأكد من وجود المجلد
    os.makedirs(base_dir, exist_ok=True)
    
    return index_path, meta_path

