"""Detect numeric recall questions copied from source without real derivation."""
from __future__ import annotations

import re

# Map Arabic-Indic digits to Latin digits
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Match integers or decimals (with thousand separators)
_NUMBER_TOKEN = re.compile(r"(?<!\w)([\d]{1,3}(?:[,،][\d]{3})+|[\d]+(?:[.][\d]+)?)(?!\w)")


def normalize_digits(text: str) -> str:
    """Normalize Arabic/Latin digits in text."""
    return str(text or "").translate(_ARABIC_DIGITS)


def extract_numbers(text: str, *, min_value: int = 4) -> set[int]:
    """Extract numbers from text (ignore 0–3 to reduce label noise)."""
    normalized = normalize_digits(text)
    found: set[int] = set()
    for match in _NUMBER_TOKEN.finditer(normalized):
        raw = match.group(1).replace(",", "").replace("،", "")
        try:
            if "." in raw:
                value = int(float(raw))
            else:
                value = int(raw)
        except ValueError:
            continue
        if value >= min_value:
            found.add(value)
    return found


def parse_primary_number(text: str) -> int | None:
    """First number appearing in text (typically the MCQ answer)."""
    normalized = normalize_digits(str(text or "").strip())
    match = _NUMBER_TOKEN.search(normalized)
    if not match:
        return None
    raw = match.group(1).replace(",", "").replace("،", "")
    try:
        if "." in raw:
            return int(float(raw))
        return int(raw)
    except ValueError:
        return None


def question_numbers(text: str) -> set[int]:
    """Numbers mentioned in the question (minimum value 4)."""
    return extract_numbers(text, min_value=4)


def looks_like_multistep_solution(solution: str) -> bool:
    """Whether the solution looks like multi-step derivation (not copied digits)."""
    sol = normalize_digits(str(solution or "").strip())
    if len(sol) < 30:
        return False
    if re.search(r"(ثم|بعد\s+ذلك|→|=>|therefore|hence|=\s*\d+\s*[×x*+\-])", sol, re.I):
        return True
    if len(re.findall(r"\d+", sol)) >= 3:
        return True
    if len(re.findall(r"[×x*+\-÷/]", sol)) >= 2:
        return True
    return False


def is_numeric_recall_from_source(item: dict, source: str) -> bool:
    """Whether the question is numeric recall: answer copied from source without derivation.

    Args:
        item: MCQ item (q, answer, solution).
        source: Source document text.

    Returns:
        True if the item should be rejected as numeric recall.
    """
    if not source.strip():
        return False

    answer = str(item.get("answer", "")).strip()
    question = str(item.get("q", "")).strip()
    solution = str(item.get("solution") or item.get("explanation") or "")

    answer_val = parse_primary_number(answer)
    if answer_val is None:
        return False

    source_nums = extract_numbers(source)
    if answer_val not in source_nums:
        return False

    if looks_like_multistep_solution(solution):
        return False

    q_nums = question_numbers(question)
    if answer_val in q_nums and len(q_nums) <= 2:
        return True

    if q_nums and q_nums.issubset(source_nums) and len(q_nums) <= 3:
        if re.search(r"[-−+×x*/]|طرح|subtract|multiply|divide|احسب|compute", question, re.I):
            return True

    if re.fullmatch(r"[\d,،.\s]+", normalize_digits(answer.strip())):
        return True

    return False
