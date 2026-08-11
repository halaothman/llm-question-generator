"""استخراج النص الخام من PDF و DOCX و TXT و MD."""
import pdfplumber
import docx
import os

def load_text(path: str) -> str:
    """
    تحميل النص من ملفات مختلفة (PDF, DOCX, TXT, MD)
    """
    ext = os.path.splitext(path)[1].lower()
    
    if ext == ".pdf":
        with pdfplumber.open(path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    
    elif ext in [".docx", ".doc"]:
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    
    elif ext in [".txt", ".md"]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")