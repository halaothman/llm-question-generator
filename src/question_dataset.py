"""
تحميل أسئلة outputs/ لصفحة المقارنة وسكربت الإحصاء.

قواعد بناء «صف التحليل» (صف واحد ≈ سؤال واحد في الإحصاء):
  • تجاهل ملفات اسمها فيه UPDATED.
  • مفتاح إزالة التكرار: (نص السؤال، LLaMA|Qwen، Vanilla|RAG، before|after).
  • استبعاد السجلات التي perplexity == 1000 (علامة فشل حساب المقياس في evals).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

OUTPUTS_DIR = Path("outputs")
METRICS = ["precision", "recall", "f1_score", "bleu", "bert_score", "perplexity"]
MODEL_DISPLAY = {"llama": "LLaMA", "qwen": "Qwen"}
METHOD_DISPLAY = {"vanilla": "Vanilla", "rag": "RAG"}


def parse_question_filename(path: Path) -> Optional[Tuple[str, str, str, str]]:
    """
    يستخرج من اسم الملف: (نموذج، طريقة، مصدر، نسخة).
    مثال: questions_qwen_rag_كتاب_new.json → Qwen, RAG, كتاب, after
    """
    stem = path.stem
    version = "after" if "_new" in stem else "before"
    parts = stem.replace("_new", "").split("_")
    if len(parts) < 3 or parts[0] != "questions":
        return None
    model, method = parts[1], parts[2]
    if model not in MODEL_DISPLAY or method not in METHOD_DISPLAY:
        return None
    source = "_".join(parts[3:]) if len(parts) >= 4 else stem
    return MODEL_DISPLAY[model], METHOD_DISPLAY[method], source.replace("_", " "), version


def list_question_files(outputs_dir: Path = OUTPUTS_DIR) -> List[Path]:
    """كل questions_*.json تحت outputs/ (بما فيها مجلدات المصادر الفرعية)."""
    if not outputs_dir.is_dir():
        return []
    return sorted(
        p
        for p in outputs_dir.rglob("questions_*.json")
        if p.is_file() and "UPDATED" not in p.stem.upper()
    )


def iter_analysis_rows(outputs_dir: Path = OUTPUTS_DIR) -> List[Dict[str, Any]]:
    """قائمة صفوف للتحليل — بدون pandas (يُعاد استخدامها في count_generated_questions)."""
    seen: set[tuple[str, str, str, str]] = set()
    rows: List[Dict[str, Any]] = []
    for path in list_question_files(outputs_dir):
        parsed = parse_question_filename(path)
        if not parsed:
            continue
        model, method, source, version = parsed
        data = json.loads(path.read_text(encoding="utf-8"))
        for q in data.get("questions", []):
            text = (q.get("question") or "").strip()
            if not text:
                continue
            key = (text, model, method, version)
            if key in seen:
                continue
            metrics = q.get("metrics") or {}
            if metrics.get("perplexity") == 1000.0:
                continue
            seen.add(key)
            row: Dict[str, Any] = {
                "file": str(path.relative_to(outputs_dir)),
                "model": model,
                "method": method,
                "source": source,
                "version": version,
            }
            for m in METRICS:
                v = metrics.get(m)
                row[m] = float(v) if isinstance(v, (int, float)) else np.nan
            ppl = row.get("perplexity")
            row["log_perplexity"] = (
                math.log(ppl) if isinstance(ppl, float) and ppl > 0 else np.nan
            )
            rows.append(row)
    return rows


def load_question_dataframe(outputs_dir: Path = OUTPUTS_DIR) -> pd.DataFrame:
    """نفس iter_analysis_rows لكن كـ DataFrame لصفحة Streamlit."""
    return pd.DataFrame(iter_analysis_rows(outputs_dir))
