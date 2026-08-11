"""تصدير أسئلة MCQ إلى DataFrame و Excel."""
from __future__ import annotations

import io

import pandas as pd

# أعمدة جدول Excel/العرض
_COLUMNS = [
    "#",
    "Type",
    "Question Kind",
    "Question",
    "Option A",
    "Option B",
    "Option C",
    "Option D",
    "Answer",
    "Solution",
]


def questions_to_dataframe(payload: dict) -> pd.DataFrame:
    """تحويل ``payload['mcq']`` إلى DataFrame بأعمدة العرض والتصدير."""
    rows: list[dict] = []

    for index, item in enumerate(payload.get("mcq", []), start=1):
        options = item.get("options", [])
        rows.append(
            {
                "#": index,
                "Type": "MCQ",
                "Question Kind": item.get("question_kind", ""),
                "Question": item.get("q", ""),
                "Option A": options[0] if len(options) > 0 else "",
                "Option B": options[1] if len(options) > 1 else "",
                "Option C": options[2] if len(options) > 2 else "",
                "Option D": options[3] if len(options) > 3 else "",
                "Answer": item.get("answer", ""),
                "Solution": item.get("solution") or item.get("explanation", ""),
            }
        )

    if not rows:
        return pd.DataFrame(columns=_COLUMNS)
    return pd.DataFrame(rows)[_COLUMNS]


def dataframe_to_excel(df: pd.DataFrame) -> bytes:
    """تصدير DataFrame إلى ملف xlsx في الذاكرة (ورقة Questions)."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Questions")
    return buffer.getvalue()
