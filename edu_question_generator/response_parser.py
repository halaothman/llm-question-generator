"""تحليل رد DeepSeek (JSON في النص) وتوحيده إلى payload داخلي."""
from __future__ import annotations

import json
import re


def _strip_model_artifacts(text: str) -> str:
    """إزالة think/redacted_thinking من رد النموذج."""
    cleaned = str(text or "")
    think_open = "<" + "think" + ">"
    think_close = "</" + "think" + ">"
    cleaned = re.sub(
        rf"{re.escape(think_open)}.*?{re.escape(think_close)}",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
    return cleaned.strip()


def _extract_json_object(text: str) -> str:
    """استخراج {…} من markdown أو نص محيط."""
    text = _strip_model_artifacts(text)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _fix_invalid_json_escapes(text: str) -> str:
    """إصلاح escape غير صالح في JSON."""
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def _remove_trailing_commas(text: str) -> str:
    """حذف فاصلة زائدة قبل } أو ]."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


def safe_json(raw: str) -> dict:
    """تحليل JSON من رد LLM مع إزالة artifacts ومحاولات إصلاح escape/فاصلة."""
    text = _extract_json_object(raw)
    candidates = [
        text,
        _fix_invalid_json_escapes(text),
        _remove_trailing_commas(text),
        _remove_trailing_commas(_fix_invalid_json_escapes(text)),
    ]
    last_error: json.JSONDecodeError | None = None
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON object found", raw, 0)


def _resolve_mcq_options(options: object) -> list[str]:
    """توحيد الخيارات (قائمة أو dict A-D)."""
    if isinstance(options, dict):
        return [str(options.get(key, "")) for key in ("A", "B", "C", "D")]
    if isinstance(options, list):
        return [str(option) for option in options[:4]]
    return []


def _resolve_mcq_answer(options: list[str], answer: object) -> str:
    """تحويل حرف A-D إلى نص الخيار إن لزم."""
    if isinstance(answer, str) and len(answer) == 1 and answer.upper() in "ABCD":
        index = "ABCD".index(answer.upper())
        if index < len(options):
            return options[index]
    return str(answer) if answer is not None else ""


def _mcq_item_from_raw(item: dict) -> dict:
    """تحويل عنصر MCQ خام من رد النموذج إلى الشكل الداخلي."""
    options = _resolve_mcq_options(item.get("options", []))
    answer = _resolve_mcq_answer(options, item.get("correct_answer", item.get("answer", "")))
    question_kind = item.get("type", item.get("question_kind", ""))
    if str(question_kind).lower() in {"mcq", "true_false", "tf", "short"}:
        question_kind = item.get("question_kind", "")
    return {
        "q": item.get("question") or item.get("q", ""),
        "options": options,
        "answer": answer,
        "solution": item.get("solution") or item.get("explanation", ""),
        "question_kind": question_kind,
    }


def normalize_payload(raw: dict) -> dict:
    """توحيد رد النموذج (mcq أو questions) إلى ``{"mcq": [...]}``."""
    mcq: list[dict] = []

    if isinstance(raw.get("mcq"), list):
        for item in raw["mcq"]:
            mcq.append(_mcq_item_from_raw(item))
        return {"mcq": mcq}

    for item in raw.get("questions", []):
        question_type = str(item.get("type", "")).lower().replace("-", "_").replace(" ", "_")
        if question_type in {
            "mcq",
            "analytical",
            "computational",
            "analysis",
            "computation",
            "application",
            "understanding",
        }:
            mcq.append(_mcq_item_from_raw(item))

    return {"mcq": mcq}


def parse_llm_mcq_response(raw: str) -> dict:
    """تحليل رد LLM كامل → payload موحّد ``{"mcq": [...]}``."""
    return normalize_payload(safe_json(raw))
