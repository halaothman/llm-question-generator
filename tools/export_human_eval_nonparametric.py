"""تصدير Mann-Whitney U و rank-biserial وتصحيح Holm من ملفات التقييم البشري CSV إلى outputs/."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nonparametric_stats import effect_magnitude, holm_bonferroni, mann_whitney_u

# --- مسارات ملفات التقييم البشري وملف النتيجة ---
OUT_PATH = ROOT / "outputs" / "human_evaluation_nonparametric.csv"

FILES = {
    "LLaMA": {
        "Vanilla": ROOT / "llama_vanilla_human_evaluation.csv",
        "RAG": ROOT / "llama_rag_human_evaluation.csv",
    },
    "Qwen": {
        "Vanilla": ROOT / "qwen_vanilla_human_evaluation.csv",
        "RAG": ROOT / "qwen_rag_human_evaluation.csv",
    },
}

# --- معايير التقييم البشري (عربي + اسم إنجليزي للتقرير) ---
METRICS = [
    ("الوضوح اللغوي", "linguistic_clarity"),
    ("الصياغة المنطقية", "logical_formulation"),
    ("الملاءمة", "relevance"),
    ("جودة الخيارات", "option_quality"),
    ("الدقة", "accuracy"),
]


def load_scores(path: Path, column: str) -> np.ndarray:
    """قراءة درجات معيار واحد من CSV مع استبعاد صف المتوسط."""
    df = pd.read_csv(path, encoding="utf-8")
    if "#" in df.columns:
        df = df[df["#"].astype(str) != "المتوسط"]
    return pd.to_numeric(df[column], errors="coerce").dropna().to_numpy()


def main() -> None:
    """مقارنة Vanilla مقابل RAG لكل نموذج ومعيار، ثم حفظ CSV."""
    rows: list[dict] = []

    for model, paths in FILES.items():
        p_raw_list: list[float] = []
        pending: list[dict] = []

        for arabic_col, criterion in METRICS:
            vanilla = load_scores(paths["Vanilla"], arabic_col)
            rag = load_scores(paths["RAG"], arabic_col)

            row: dict = {
                "model": model,
                "criterion": criterion,
                "criterion_ar": arabic_col,
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

        adjusted = holm_bonferroni(p_raw_list).tolist()
        for row, p_holm in zip(pending, adjusted):
            row["p_holm"] = round(p_holm, 4)
            row["significant_after_holm"] = "Yes" if p_holm < 0.05 else "No"
            rows.append(row)

    out_df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved {len(out_df)} rows to {OUT_PATH}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
