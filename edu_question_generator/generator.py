"""Generate MCQ questions via DeepSeek: prompts, sanitization, and model calls."""
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

# Characters forbidden in question text (CJK, Cyrillic, etc.)
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
    """Detect document language (ar/en) via langdetect; defaults to en on failure."""
    try:
        return "ar" if detect(text) == "ar" else "en"
    except Exception:
        return "en"


def sanitize_text(text: str) -> str:
    """Strip CJK/Cyrillic/Devanagari and similar characters from question or solution text."""
    if not text:
        return text
    cleaned = FORBIDDEN_CHARS.sub("", str(text))
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def sanitize_payload(payload: dict) -> dict:
    """Sanitize every MCQ field (q, options, answer, solution, question_kind)."""
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
    """Generate MCQs from a single segment via DeepSeek with sanitization and JSON parsing.

    Args:
        context: Segment text to generate questions from.
        lang: Document language (ar/en) for system message selection.
        num_questions: Desired count; None = up to 3 questions.
        model: Model identifier; empty = DEEPSEEK_MODEL.
        api_key: DeepSeek API key.

    Returns:
        dict with ``mcq`` key containing sanitized question list.

    Raises:
        json.JSONDecodeError: If JSON parsing fails after two attempts.
    """
    prompt = build_deepseek_prompt(context, num_questions)
    system = build_deepseek_system_message(lang)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    resolved_model = model or DEEPSEEK_MODEL
    last_error: json.JSONDecodeError | None = None
    # Retry once on JSON parse failure (model may fix format on second attempt)
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
