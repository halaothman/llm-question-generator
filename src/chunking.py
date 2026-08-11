"""تقسيم النص إلى قطع متداخلة بحجم وتداخل قابل للضبط."""
def chunk_text(text: str, size: int = 500, overlap: int = 100):
    """
    تقسيم النص إلى قطع متداخلة
    
    Args:
        text: النص المراد تقسيمه
        size: حجم كل قطعة
        overlap: مقدار التداخل بين القطع
    
    Returns:
        list: قائمة بالقطع
    """
    chunks = []
    i = 0
    
    while i < len(text):
        chunk = text[i:i + size]
        if chunk.strip():  # تجاهل القطع الفارغة
            chunks.append(chunk)
        i += size - overlap
    
    return chunks