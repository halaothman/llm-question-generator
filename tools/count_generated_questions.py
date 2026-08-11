"""
إحصاء عدد الأسئلة المولّدة من ملفات outputs/questions_*.json.

يُطبع رقمان مهمّان:
  • خام (raw): مجموع len(questions) في كل ملف — قد يتكرر نفس السؤال في ملفات مختلفة.
  • صفوف التحليل: بعد إزالة التكرار واستبعاد perplexity=1000 — نفس منطق صفحة «المقارنة والتحليل».
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.question_counts import count_questions_in_json_file
from src.question_dataset import iter_analysis_rows, list_question_files

OUTPUTS = ROOT / "outputs"


def main() -> int:
    # كل ملفات questions_<model>_<method>_<source>[_new].json تحت outputs/
    files = list_question_files(OUTPUTS)
    if not files:
        print(f"لا توجد ملفات questions_*.json تحت: {OUTPUTS}")
        return 1

    # العدّ الخام: كل سجل في questions[] يُحسب مرة — حتى لو تكرر نص السؤال في ملف آخر
    raw_total = sum(count_questions_in_json_file(p) for p in files)

    # صف واحد لكل سؤال فريد (نص + نموذج + طريقة + before|after)، مع فلترة فشل المقاييس
    rows = iter_analysis_rows(OUTPUTS)

    by_file = {str(p.relative_to(OUTPUTS)): count_questions_in_json_file(p) for p in files}
    # before = ملف بدون _new (قبل التحسين)، after = اسم ينتهي بـ _new.json
    by_model_method_version = Counter(
        (r["model"], r["method"], r["version"]) for r in rows
    )

    print("=== إحصاء الأسئلة المولّدة ===")
    print(f"المجلد: {OUTPUTS}")
    print(f"عدد ملفات JSON: {len(files)}")
    print(f"إجمالي الأسئلة (خام، مع تكرار بين الملفات): {raw_total}")
    print()
    print("--- بعد إزالة التكرار (كصفحة المقارنة) ---")
    print(f"صفوف التحليل: {len(rows)}")
    print()
    print("--- حسب النموذج / Vanilla|RAG / before|after ---")
    for key in sorted(by_model_method_version.keys()):
        print(f"  {key[0]:5}  {key[1]:7}  {key[2]:6}  ->  {by_model_method_version[key]}")
    print()
    print("--- حسب الملف ---")
    for name, n in sorted(by_file.items()):
        print(f"  {n:4}  {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
