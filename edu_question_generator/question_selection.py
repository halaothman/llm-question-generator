"""اختيار أسئلة MCQ: توزيع الميزانية، إزالة التكرار، وحد أقصى مع حصص نوعية."""
from __future__ import annotations

import re

# أنواع MCQ حسب حقل type من JSON النموذج
_COMPUTATION_TYPES = frozenset(
    {"computation", "computational", "حساب", "حسابي", "calculation", "numeric"}
)

_ANALYSIS_APPLICATION_TYPES = frozenset(
    {"analysis", "application", "تحليل", "تطبيق", "analytical", "applied"}
)


def distribute_question_counts(num_questions: int, segment_count: int) -> list[int]:
    """توزيع ميزانية الأسئلة على مقاطع المستند (باقي → المقاطع الأولى)."""
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
    """مفتاح إزالة تكرار MCQ (نص السؤال + الخيارات)."""
    question = re.sub(r"\s+", " ", str(item.get("q", "")).strip().lower())
    options = "|".join(
        re.sub(r"\s+", " ", str(option).strip().lower())
        for option in item.get("options", [])
    )
    return f"{question}::{options}"


def dedupe_mcq(items: list[dict]) -> list[dict]:
    """إزالة أسئلة MCQ مكررة."""
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
    """نوع السؤال من حقل type أو question_kind."""
    raw = str(item.get("question_kind") or item.get("type") or "").strip().lower()
    return raw.replace("-", "_").replace(" ", "_")


def _is_computation(item: dict) -> bool:
    """هل السؤال من نوع حساب (computation) حسب حقل type؟"""
    return _question_type(item) in _COMPUTATION_TYPES


def _is_analysis_or_application(item: dict) -> bool:
    """هل السؤال من نوع تحليل أو تطبيق حسب حقل type؟"""
    return _question_type(item) in _ANALYSIS_APPLICATION_TYPES


def cap_and_diversify_mcq(
    items: list[dict],
    *,
    max_total: int,
    min_computation: int,
    min_analysis_application: int,
) -> list[dict]:
    """اختيار MCQ نهائي: حصة تحليل/تطبيق، ثم حساب، ثم الباقي — مع dedupe.

    Args:
        items: قائمة الأسئلة بعد التحقق.
        max_total: العدد الأقصى المطلوب (مثلاً 20).
        min_computation: حد أدنى لأسئلة الحساب.
        min_analysis_application: حد أدنى لأسئلة التحليل/التطبيق.
    """
    if max_total <= 0 or not items:
        return []

    selected: list[dict] = []
    seen: set[str] = set()

    def try_add(item: dict) -> bool:
        """إضافة سؤال إن لم يُتجاوز الحد الأقصى ولم يُكرّر مفتاحه."""
        if len(selected) >= max_total:
            return False
        key = question_key(item)
        if not key or key in seen:
            return False
        seen.add(key)
        selected.append(item)
        return True

    for item in items:
        if sum(1 for x in selected if _is_analysis_or_application(x)) >= min_analysis_application:
            break
        if _is_analysis_or_application(item):
            try_add(item)

    for item in items:
        if sum(1 for x in selected if _is_computation(x)) >= min_computation:
            break
        if _is_computation(item):
            try_add(item)

    for item in items:
        try_add(item)

    return selected[:max_total]
