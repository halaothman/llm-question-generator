"""تنظيف النص العربي و stemming لتحسين الفهرسة والاسترجاع."""
import re
import unicodedata
from nltk.stem.snowball import SnowballStemmer

# تهيئة الـ stemmer العربي
_ar_stem = SnowballStemmer("arabic")

# أنماط regex للتنظيف
_AR_DIAC = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")  # التشكيل
_PUNCT = re.compile(r"[^\w\s\u0600-\u06FF]")  # الرموز (إبقاء الحروف العربية)

def clean_ar(text: str) -> str:
    """
    تنظيف النص العربي من التشكيل والرموز
    
    Args:
        text: النص المراد تنظيفه
    
    Returns:
        str: النص المنظف
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    # إزالة التشكيل
    text = _AR_DIAC.sub("", text)
    
    # إزالة الرموز (إبقاء الحروف العربية والأرقام والمسافات)
    text = _PUNCT.sub(" ", text)
    
    # تنظيف المسافات المتعددة
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def stem_ar(text: str) -> str:
    """
    تجذيع الكلمات العربية
    
    Args:
        text: النص المراد تجذيعه
    
    Returns:
        str: النص المجذوع
    """
    if not text:
        return ""
    
    try:
        words = text.split()
        stemmed_words = []
        
        for word in words:
            if word.strip():
                stemmed_words.append(_ar_stem.stem(word))
        
        return " ".join(stemmed_words)
    except Exception:
        # في حالة فشل التجذيع، إرجاع النص الأصلي
        return text