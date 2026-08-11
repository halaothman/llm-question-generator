"""MCQ selection: budget distribution, deduplication, and capped type quotas."""
from __future__ import annotations

import re

# MCQ types from the model JSON ``type`` field
_COMPUTATION_TYPES = frozenset(
    {"computation", "computational", "حساب", "حسابي", "calculation", "numeric"}
)

_ANALYSIS_APPLICATION_TYPES = frozenset(
    {"analysis", "application", "تحليل", "تطبيق", "analytical", "applied"}
)


def distribute_question_counts(num_questions: int, segment_count: int) -> list[int]:
    """Distribute question budget across document segments (remainder → first segments)."""
    if segment_count <= 0:
        return []
    if num_questions <= 0:
        return [0] * segment_count
    if num_questions <= segment_count:
        return [1 if index < num_questions else 0 for index in range(segment_count)]
    base = num_questions // segment_count
    remainder = num_questions % segment_count
    return [base + (1 if index < remainder else 0) for index in range(segment_count)]


def question_key(item: dict) -> str:
    """Dedup key for MCQ items (question text + options)."""
    question = re.sub(r"\s+", " ", str(item.get("q", "")).strip().lower())
    options = "|".join(
        re.sub(r"\s+", " ", str(option).strip().lower())
        for option in item.get("options", [])
    )
    return f"{question}::{options}"


def dedupe_mcq(items: list[dict]) -> list[dict]:
    """Remove duplicate MCQ items."""
    deduped: list[dict] = []
    seen: set[str] = set()
    for item in items:
        key = question_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _question_type(item: dict) -> str:
    """Question type from ``type`` or ``question_kind`` field."""
    raw = str(item.get("question_kind") or item.get("type") or "").strip().lower()
    return raw.replace("-", "_").replace(" ", "_")


def _is_computation(item: dict) -> bool:
    """Whether the item is a computation-type question."""
    return _question_type(item) in _COMPUTATION_TYPES


def _is_analysis_or_application(item: dict) -> bool:
    """Whether the item is analysis or application type."""
    return _question_type(item) in _ANALYSIS_APPLICATION_TYPES


def cap_and_diversify_mcq(
    items: list[dict],
    *,
    max_total: int,
    min_computation: int,
    min_analysis_application: int,
) -> list[dict]:
    """Select final MCQs: analysis/application quota, then computation, then fill — with dedupe.

    Args:
        items: Validated question list.
        max_total: Maximum desired count (e.g. 20).
        min_computation: Minimum computation questions.
        min_analysis_application: Minimum analysis/application questions.
    """
    if max_total <= 0 or not items:
        return []

    selected: list[dict] = []
    seen: set[str] = set()

    def try_add(item: dict) -> bool:
        """Add item if under cap and not already seen."""
        if len(selected) >= max_total:
            return False
        key = question_key(item)
        if not key or key in seen:
            return False
        seen.add(key)
        selected.append(item)
        return True

    # Pass 1: meet analysis/application minimum
    for item in items:
        if sum(1 for x in selected if _is_analysis_or_application(x)) >= min_analysis_application:
            break
        if _is_analysis_or_application(item):
            try_add(item)

    # Pass 2: meet computation minimum
    for item in items:
        if sum(1 for x in selected if _is_computation(x)) >= min_computation:
            break
        if _is_computation(item):
            try_add(item)

    # Pass 3: fill remaining slots in original order
    for item in items:
        try_add(item)

    return selected[:max_total]
