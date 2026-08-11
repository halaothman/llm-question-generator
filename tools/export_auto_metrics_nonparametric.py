"""
تصدير نتائج Mann-Whitney U و rank-biserial وتصحيح Holm للمقاييس الآلية.

يقرأ كل questions_*.json تحت outputs/، يجمع مقاييس كل سؤال، ثم يقارن Vanilla
مقابل RAG بثلاث صيغ (baseline فقط، improved فقط، أو الاثنان معاً كـ RAG-all).

المخرجات: outputs/automatic_evaluation_nonparametric.csv
(مناسب للجداول في الرسالة؛ صفحة Streamlit «المقارنة والتحليل» تستخدم منطقاً
قريباً لكنها تقسم before/after بدل RAG-baseline vs RAG-improved.)

تشغيل:
    python tools/export_auto_metrics_nonparametric.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nonparametric_stats import effect_magnitude, holm_bonferroni, mann_whitney_u

# --- مسارات المشروع وملف الإخراج ---
OUTPUTS = ROOT / "outputs"
OUT_PATH = OUTPUTS / "automatic_evaluation_nonparametric.csv"

# --- المقاييس المحسوبة عند الحفظ (src/evals.calculate_all_metrics) ---
METRICS = ["precision", "recall", "f1_score", "bleu", "bert_score", "perplexity"]
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1",
    "bleu": "BLEU",
    "bert_score": "BERTScore",
    "perplexity": "Perplexity",
}

# (عنوان في CSV، تسمية Vanilla، تسمية مجموعة RAG في DataFrame)
COMPARISONS = [
    ("Vanilla vs RAG-improved", "Vanilla", "RAG-improved"),
    ("Vanilla vs RAG-baseline", "Vanilla", "RAG-baseline"),
    ("Vanilla vs RAG-all", "Vanilla", "RAG-all"),
]


def rag_condition(path: Path) -> str | None:
    """
    استنتاج «شرط التجربة» من اسم الملف فقط (لا يستخدم before/after كصفحة 3).

    questions_*_vanilla_*.json  → Vanilla
    questions_*_rag_*_new.json    → RAG-improved (توليد بعد التحسين)
    questions_*_rag_*.json        → RAG-baseline (بدون لاحقة _new)
    """
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    method = parts[2]
    if method == "vanilla":
        return "Vanilla"
    if method == "rag":
        return "RAG-improved" if path.name.endswith("_new.json") else "RAG-baseline"
    return None


def load_question_metrics() -> pd.DataFrame:
    """
    صف واحد لكل سؤال في كل ملف: model + condition + المقاييس الستة.

    يُستبعد السؤال إذا perplexity غير صالحة (0 أو 1000 أو مفقودة) — نفس فكرة
    استبعاد فشل حساب Perplexity في التحليل الآخر.
    """
    rows = []
    for path in OUTPUTS.rglob("questions_*.json"):
        if "UPDATED" in path.stem.upper():
            continue
        cond = rag_condition(path)
        if cond is None:
            continue
        # parts[1] = llama | qwen من questions_<model>_<method>_...
        model = path.stem.split("_")[1]
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        for q in data.get("questions", []):
            m = q.get("metrics") or {}
            ppl = m.get("perplexity")
            if ppl in (0.0, 1000.0, None):
                continue
            row = {
                "model": {"llama": "LLaMA", "qwen": "Qwen"}.get(model, model),
                "condition": cond,
            }
            for key in METRICS:
                v = m.get(key)
                row[key] = float(v) if isinstance(v, (int, float)) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def add_rag_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    نسخ صفوف RAG-baseline و RAG-improved مع condition = RAG-all
    لإجراء مقارنة Vanilla مقابل «كل أسئلة RAG» في تجربة واحدة.
    """
    rag = df[df["condition"].isin(["RAG-baseline", "RAG-improved"])].copy()
    rag["condition"] = "RAG-all"
    return pd.concat([df, rag], ignore_index=True)


def compare_group(
    df: pd.DataFrame,
    model: str,
    comparison: str,
    vanilla_label: str,
    rag_label: str,
) -> list[dict]:
    """
    Mann-Whitney بين مجموعتين (Vanilla vs RAG) لكل مقياس من METRICS.

    Holm-Bonferroni يُطبَّق على الـ p-values الستة داخل هذا (model + comparison)
    فقط — وليس على كل الصفوف في الملف دفعة واحدة.
    """
    sub = df[df["model"] == model]
    rows: list[dict] = []
    p_raw_list: list[float] = []
    pending: list[dict] = []

    for metric in METRICS:
        vanilla = sub.loc[sub["condition"] == vanilla_label, metric].dropna().to_numpy()
        rag = sub.loc[sub["condition"] == rag_label, metric].dropna().to_numpy()

        row = {
            "model": model,
            "comparison": comparison,
            "metric": metric,
            "metric_label": METRIC_LABELS[metric],
            "n_vanilla": int(len(vanilla)),
            "n_rag": int(len(rag)),
            "mean_vanilla": round(float(np.mean(vanilla)), 4) if len(vanilla) else None,
            "mean_rag": round(float(np.mean(rag)), 4) if len(rag) else None,
            "median_vanilla": round(float(np.median(vanilla)), 4) if len(vanilla) else None,
            "median_rag": round(float(np.median(rag)), 4) if len(rag) else None,
            "mannwhitney_u": None,
            "p_raw": None,
            "p_holm": None,
            "rank_biserial_r_rb": None,
            "effect_magnitude": None,
            "significant_after_holm": None,
        }

        # Mann-Whitney يحتاج عينة كافية؛ أقل من 3 → لا اختبار
        if len(vanilla) < 3 or len(rag) < 3:
            row["effect_magnitude"] = "Insufficient sample"
            rows.append(row)
            continue

        mw = mann_whitney_u(vanilla, rag)
        assert mw is not None
        row.update(
            {
                "mannwhitney_u": round(mw["u"], 4),
                "p_raw": round(mw["p"], 4),
                "rank_biserial_r_rb": round(mw["rb"], 4),
                "effect_magnitude": effect_magnitude(mw["rb"]),
            }
        )
        p_raw_list.append(mw["p"])
        pending.append(row)

    # تصحيح تعدد الاختبارات: 6 مقاييس × (نموذج × مقارنة) = دفعة Holm منفصلة
    adjusted = holm_bonferroni(p_raw_list).tolist()
    for row, p_holm in zip(pending, adjusted):
        row["p_holm"] = round(p_holm, 4)
        row["significant_after_holm"] = "Yes" if p_holm < 0.05 else "No"
        rows.append(row)

    return rows


def main() -> None:
    """تحميل البيانات، 2×3 مقارنات (LLaMA/Qwen × COMPARISONS)، حفظ CSV وطباعة ملخص."""
    df = load_question_metrics()
    if df.empty:
        raise SystemExit("No automatic metric data found in outputs/.")

    df = add_rag_all(df)
    all_rows: list[dict] = []

    for model in ("LLaMA", "Qwen"):
        for comparison, vanilla_label, rag_label in COMPARISONS:
            all_rows.extend(compare_group(df, model, comparison, vanilla_label, rag_label))

    out_df = pd.DataFrame(all_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved {len(out_df)} rows to {OUT_PATH}\n")
    for comparison in [c[0] for c in COMPARISONS]:
        print(f"=== {comparison} ===")
        part = out_df[out_df["comparison"] == comparison]
        print(
            part[
                [
                    "model",
                    "metric_label",
                    "mean_vanilla",
                    "mean_rag",
                    "p_raw",
                    "p_holm",
                    "rank_biserial_r_rb",
                    "significant_after_holm",
                ]
            ].to_string(index=False)
        )
        print()


if __name__ == "__main__":
    main()
