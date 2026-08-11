"""توليد أسئلة MCQ عبر DeepSeek: prompts، تنظيف، واستدعاء النموذج."""
from __future__ import annotations

import json
import re
from typing import Literal

from langdetect import detect

from .config import DEEPSEEK_MODEL
from .llm_client import chat_complete
from .prompts.deepseek import build_deepseek_prompt, build_deepseek_system_message
from .response_parser import parse_llm_mcq_response

Lang = Literal["ar", "en"]

# رموز غير مسموحة في نص السؤال (CJK، Cyrillic، …)
FORBIDDEN_CHARS = re.compile(
    "["
    "\u4e00-\u9fff"  # CJK Unified Ideographs
    "\u3400-\u4dbf"  # CJK Extension A
    "\u3040-\u30ff"  # Hiragana + Katakana
    "\u31f0-\u31ff"  # Katakana extensions
    "\uac00-\ud7af"  # Hangul syllables
    "\u1100-\u11ff"  # Hangul jamo
    "\u0400-\u04ff"  # Cyrillic
    "\u0900-\u097f"  # Devanagari
    "\u0980-\u09ff"  # Bengali
    "\u0e00-\u0e7f"  # Thai
    "]+"
)


def detect_lang(text: str) -> Lang:
    """كشف لغة المستند (ar/en) عبر langdetect؛ الافتراضي en عند الفشل."""
    try:
        return "ar" if detect(text) == "ar" else "en"
    except Exception:
        return "en"


def sanitize_text(text: str) -> str:
    """إزالة رموز CJK/Cyrillic/Devanagari وغيرها من نص السؤال أو الحل."""
    if not text:
        return text
    cleaned = FORBIDDEN_CHARS.sub("", str(text))
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def sanitize_payload(payload: dict) -> dict:
    """تنظيف حقول كل عنصر MCQ (q, options, answer, solution, question_kind)."""
    for item in payload.get("mcq", []):
        item["q"] = sanitize_text(item.get("q", ""))
        item["solution"] = sanitize_text(item.get("solution") or item.get("explanation", ""))
        item["explanation"] = item["solution"]
        if item.get("question_kind"):
            item["question_kind"] = sanitize_text(str(item["question_kind"]))
        item["options"] = [sanitize_text(option) for option in item.get("options", [])]
        if not isinstance(item.get("answer"), bool):
            item["answer"] = sanitize_text(str(item.get("answer", "")))
    return payload


def generate_questions(
    context: str,
    lang: Lang,
    num_questions: int | None = None,
    model: str = "",
    api_key: str | None = None,
) -> dict:
    """توليد MCQ من مقطع واحد عبر DeepSeek مع تنظيف وتحليل JSON.

    Args:
        context: نص المقطع المراد توليد أسئلة منه.
        lang: لغة المستند (ar/en) لاختيار رسالة system.
        num_questions: العدد المطلوب؛ None = حتى 3 أسئلة.
        model: معرّف النموذج؛ فارغ = DEEPSEEK_MODEL.
        api_key: مفتاح DeepSeek.

    Returns:
        dict بمفتاح ``mcq`` يحتوي قائمة أسئلة منظّفة.

    Raises:
        json.JSONDecodeError: إذا فشل تحليل JSON بعد محاولتين.
    """
    prompt = build_deepseek_prompt(context, num_questions)
    system = build_deepseek_system_message(lang)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    resolved_model = model or DEEPSEEK_MODEL
    last_error: json.JSONDecodeError | None = None
    for attempt in range(2):
        try:
            content = chat_complete(resolved_model, messages, api_key=api_key)
            return sanitize_payload(parse_llm_mcq_response(content))
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt == 0:
                continue
            raise
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON object found", "", 0)
