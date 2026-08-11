"""التحقق من أسئلة MCQ: بنية MCQ محلياً + جودة/إسناد عبر LLM."""
from __future__ import annotations

import json
import re

from .response_parser import safe_json
from .llm_client import chat_complete
from .numeric_recall import is_numeric_recall_from_source

VALIDATION_CONTEXT_LIMIT = 14_000
MAX_SOLUTION_CHARS = 500
MIN_QUESTION_CHARS = 10


def _normalize(text: str) -> str:
    """تطبيع للمقارنة (مسافات + lowercase)."""
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _passes_structural_checks(item: dict, source: str = "") -> bool:
    """فحص شكل MCQ محلياً: 4 خيارات فريدة، إجابة ضمن الخيارات، ورفض recall رقمي."""
    question = str(item.get("q", "")).strip()
    options = [str(option).strip() for option in item.get("options", []) if str(option).strip()]
    answer = str(item.get("answer", "")).strip()

    if len(question) < MIN_QUESTION_CHARS:
        return False
    if len(options) != 4:
        return False
    if len({_normalize(option) for option in options}) < 4:
        return False
    if not answer:
        return False
    if _normalize(answer) not in [_normalize(option) for option in options]:
        return False

    solution = str(item.get("solution") or item.get("explanation") or "")
    if len(solution) > MAX_SOLUTION_CHARS:
        return False

    if source and is_numeric_recall_from_source(item, source):
        return False

    return True


def _build_validation_prompt(source: str, questions: list[dict]) -> str:
    """prompt التحقق LLM: قائمة ids للإبقاء/الرفض."""
    compact_questions = [
        {
            "id": index + 1,
            "type": item.get("question_kind", item.get("type", "")),
            "q": item.get("q", ""),
            "options": item.get("options", []),
            "answer": item.get("answer", ""),
            "solution": item.get("solution", item.get("explanation", "")),
        }
        for index, item in enumerate(questions)
    ]

    return f"""أنت أستاذ جامعي صارم تتحقق من أسئلة MCQ عربية صعبة.

المستند المصدر (الدليل الوحيد المسموح):
{source[:VALIDATION_CONTEXT_LIMIT]}

الأسئلة للتحقق:
{json.dumps(compact_questions, ensure_ascii=False)}

ارفض السؤال إذا تحقق أي شرط:

الإسناد للمصدر:
1. لا يمكن الإجابة عليه بالكامل من المستند فقط.
2. لا يمكن التحقق من الإجابة الصحيحة من المصدر وحده.
3. يعتمد على افتراضات مخفية («إذا افترضنا»، «لنفترض»، بيانات غير مذكورة).
4. يستخدم معرفة خارجية غير موجودة في المصدر.
5. يختلق معادلات أو أرقام أو أبعاد tensors أو أسماء ملفات أو معماريات أو hyperparameters.

جودة السؤال:
6. حفظ/recall بسيط بدل تحليل أو حساب.
7. يُجاب بقراءة سطر واحد من الكود أو المستند («ما قيمة…»، «ما هو…»، «كم عدد العناصر»).
8. يسأل مباشرة عن أسماء متغيرات أو ملفات أو imports أو قيم ابتدائية أو epochs دون تفكير حقيقي.
9. سؤال تعريف سطحي («ما هو X؟»، «عرّف Y») بما فيها أدوات ML (TensorFlow، SGD، Adam، Dropout، …).
10. سؤال طرح/عد ميكانيكي (مثل: كم كلمة تُتجاهل max_words؟).
11. غير صعب بما يكفي لامتحان جامعي.
12. يستخدم كلمات غامضة غير موجودة في المصدر (الأفضل، عادة، ربما، من المحتمل، …).

حسابي / تحليلي:
13. حسابي: لا يمكن حساب الإجابة من القيم في السؤال أو المصدر.
14. تحليلي: تعريف بسيط أو سؤال غامض أو ذاتي.

الخيارات / الشرح:
15. خيارات خاطئة غير منطقية أو هراء أو distractors عامة («تقليل الصور»، «تخزين عميق»، …).
16. أكثر من خيار يمكن أن يكون صحيحاً بشكل معقول.
17. الحل يكرر الإجابة الصحيحة فقط أو يبدو غير واثق («ربما»، «من المحتمل»، «غير واثق»).

عند الشك، ارفض.

أعد JSON فقط:
{{
  "keep": [1, 2],
  "rejected": [{{"id": 3, "reason": "..."}}]
}}

«keep» يجب أن تسرد ids الأسئلة الصالحة فقط."""


def _llm_filter_ids(
    source: str,
    questions: list[dict],
    api_key: str | None,
    model: str,
) -> set[int]:
    """استدعاء LLM وإرجاع مجموعة ids المقبولة."""
    if not questions:
        return set()

    prompt = _build_validation_prompt(source, questions)
    system = "تحقق من أسئلة MCQ مقابل المستند. كن صارماً. JSON فقط."
    content = chat_complete(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        api_key=api_key,
        max_tokens=2048,
    )
    parsed = safe_json(content)
    keep_ids: set[int] = set()
    for item in parsed.get("keep", []):
        try:
            keep_ids.add(int(item))
        except (TypeError, ValueError):
            continue
    return keep_ids


def filter_mcq_payload(
    payload: dict,
    source: str,
    api_key: str | None,
    model: str = "",
) -> dict:
    """فلترة MCQ: فحص بنية محلي ثم قبول/رفض عبر LLM مقابل المصدر.

    Args:
        payload: dict يحتوي ``mcq``.
        source: نص المستند الأصلي للتحقق من الإسناد.
        api_key: مفتاح DeepSeek للتحقق.
        model: معرّف النموذج.

    Returns:
        payload مُحدَّث بقائمة mcq المقبولة فقط.
    """
    mcq_items = payload.get("mcq", [])
    if not mcq_items:
        return payload

    structurally_valid = [item for item in mcq_items if _passes_structural_checks(item, source)]
    if not structurally_valid:
        payload["mcq"] = []
        return payload

    try:
        keep_ids = _llm_filter_ids(source, structurally_valid, api_key, model)
    except Exception:
        payload["mcq"] = structurally_valid
        return payload

    payload["mcq"] = [
        item for index, item in enumerate(structurally_valid, start=1) if index in keep_ids
    ]
    return payload
