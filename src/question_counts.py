"""
عدّ الأسئلة — مصدران:

1) مخرجات Ollama مباشرة: dict فيه mcq[] و tf[] (قبل الحفظ في JSON).
2) ملفات محفوظة: قائمة questions[] أو metadata.total_questions.

العدّ هنا يتجاهل العناصر الفارغة (بدون نص سؤال)، بما يطابق save_questions_separate_file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple

QuestionItem = Tuple[str, Dict[str, Any]]


def question_text(item: dict) -> str:
    """النموذج قد يستخدم q أو question."""
    return (item.get("q") or item.get("question") or "").strip()


def iter_llm_questions(questions_data: Dict[str, Any]) -> Iterator[QuestionItem]:
    """(نوع, عنصر) لكل سؤال MCQ أو صح/خطأ له نص."""
    for qtype in ("mcq", "tf"):
        for item in questions_data.get(qtype) or []:
            if isinstance(item, dict) and question_text(item):
                yield qtype, item


def count_llm_questions(questions_data: Dict[str, Any]) -> int:
    """إجمالي MCQ + TF الصالحة — يُستخدم في واجهة التوليد و metadata.total_questions."""
    return sum(1 for _ in iter_llm_questions(questions_data))


def count_saved_questions(data: Dict[str, Any]) -> int:
    """من JSON محفوظ: يفضّل len(questions) ثم يرجع إلى total_questions في metadata."""
    questions = data.get("questions")
    if isinstance(questions, list):
        return len(questions)
    meta = data.get("metadata") or {}
    total = meta.get("total_questions")
    return int(total) if isinstance(total, int) else 0


def count_questions_in_json_file(path: Path) -> int:
    """عدّ خام لملف واحد (بدون إزالة تكرار بين الملفات)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    return count_saved_questions(data)


def sum_questions_in_files(paths: Iterable[Path]) -> int:
    """مجموع count_questions_in_json_file على عدة مسارات."""
    return sum(count_questions_in_json_file(p) for p in paths)
