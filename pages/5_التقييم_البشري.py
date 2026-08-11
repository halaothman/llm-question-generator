"""صفحة عرض نتائج التقييم البشري: جداول Mann-Whitney وShapiro وHolm."""
import sys
from pathlib import Path
from typing import Dict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from src.nonparametric_stats import (
    apply_holm_to_rows,
    effect_magnitude,
    format_p_value,
    mann_whitney_u,
    shapiro_normality_label,
    shapiro_wilk,
)
from src.ui_styles import inject_app_styles

st.set_page_config(page_title="التقييم البشري", layout="wide")

inject_app_styles()

HUMAN_METRICS = ["الوضوح اللغوي", "الصياغة المنطقية", "الملاءمة", "جودة الخيارات", "الدقة"]

# أعمدة تُحسب لكن لا تُعرض في جدول Mann-Whitney (النموذج من عنوان الجدول)
MANN_WHITNEY_HIDDEN_COLS = ("النموذج",)


def mann_whitney_table(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in MANN_WHITNEY_HIDDEN_COLS if c in df.columns]
    return df.drop(columns=drop) if drop else df


def get_clean_human_df(data: dict, key: str) -> pd.DataFrame:
    """إزالة صف المتوسط وتحويل المقاييس البشرية إلى أرقام."""
    df = data[key]["df"].copy()
    df = df[df["#"].astype(str) != "المتوسط"].copy()
    for metric in HUMAN_METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def build_human_statistical_results(data: dict):
    """حساب نتائج Mann-Whitney مع Holm وShapiro-Wilk."""
    required = ["llama_vanilla", "llama_rag", "qwen_vanilla", "qwen_rag"]
    if not all(key in data for key in required):
        return None, None

    groups = {
        "LLaMA": {
            "Vanilla": get_clean_human_df(data, "llama_vanilla"),
            "RAG": get_clean_human_df(data, "llama_rag"),
        },
        "Qwen": {
            "Vanilla": get_clean_human_df(data, "qwen_vanilla"),
            "RAG": get_clean_human_df(data, "qwen_rag"),
        },
    }

    detail_rows = []
    shapiro_rows = []

    for model, model_groups in groups.items():
        model_rows = []

        for metric in HUMAN_METRICS:
            vanilla = model_groups["Vanilla"][metric].dropna().to_numpy()
            rag = model_groups["RAG"][metric].dropna().to_numpy()

            for method_name, values in (("Vanilla", vanilla), ("RAG", rag)):
                sw = shapiro_wilk(values)
                if sw["W"] is not None:
                    shapiro_rows.append(
                        {
                            "النموذج": model,
                            "الطريقة": method_name,
                            "المقياس": metric,
                            "حجم العينة": sw["n"],
                            "W": sw["W"],
                            "p": format_p_value(sw["p"]),
                            "الاستنتاج": shapiro_normality_label(sw["p"]),
                        }
                    )

            mw = mann_whitney_u(vanilla, rag)
            if mw is None:
                model_rows.append(
                    {
                        "النموذج": model,
                        "المقياس": metric,
                        "n (Vanilla)": len(vanilla),
                        "n (RAG)": len(rag),
                        "وسيط Vanilla": None,
                        "وسيط RAG": None,
                        "U": None,
                        "p (قبل Holm)": None,
                        "p بعد Holm": None,
                        "Rank-Biserial": None,
                        "حجم الأثر": "عينة غير كافية",
                        "القرار": "غير كافٍ",
                    }
                )
                continue

            model_rows.append(
                {
                    "النموذج": model,
                    "المقياس": metric,
                    "n (Vanilla)": mw["n_a"],
                    "n (RAG)": mw["n_b"],
                    "وسيط Vanilla": round(mw["median_a"], 2),
                    "وسيط RAG": round(mw["median_b"], 2),
                    "U": round(mw["u"], 2),
                    "p (قبل Holm)": round(mw["p"], 4),
                    "p_raw": mw["p"],
                    "p بعد Holm": None,
                    "Rank-Biserial": round(mw["rb"], 4),
                    "حجم الأثر": effect_magnitude(mw["rb"]),
                    "القرار": None,
                }
            )

        apply_holm_to_rows(
            model_rows,
            holm_key="p بعد Holm",
            decision_key="القرار",
        )

        detail_rows.extend(model_rows)

    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty and "p_raw" in detail_df.columns:
        detail_df = detail_df.drop(columns=["p_raw"])

    return detail_df, pd.DataFrame(shapiro_rows)


def load_human_evaluation_data() -> Dict[str, dict]:
    """تحميل بيانات التقييم البشري من CSV."""
    files = {
        "llama_vanilla": "llama_vanilla_human_evaluation.csv",
        "llama_rag": "llama_rag_human_evaluation.csv",
        "qwen_vanilla": "qwen_vanilla_human_evaluation.csv",
        "qwen_rag": "qwen_rag_human_evaluation.csv",
    }

    data = {}
    for key, filename in files.items():
        file_path = Path(filename)
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, encoding="utf-8-sig")
                avg_row = df[df["#"].astype(str) == "المتوسط"]
                if not avg_row.empty:
                    data[key] = {"df": df, "averages": avg_row.iloc[0].to_dict()}
            except Exception as e:
                st.error(f"خطأ في قراءة {filename}: {e}")
        else:
            st.warning(f"الملف غير موجود: {filename}")

    return data


def main():
    st.title("التقييم البشري")

    data = load_human_evaluation_data()

    if not data:
        st.error("لم يتم العثور على ملفات التقييم البشري!")
        st.info(
            "**الملفات المطلوبة:**\n"
            "- `llama_vanilla_human_evaluation.csv`\n"
            "- `llama_rag_human_evaluation.csv`\n"
            "- `qwen_vanilla_human_evaluation.csv`\n"
            "- `qwen_rag_human_evaluation.csv`"
        )
        return

    if len(data) < 4:
        st.warning(f"تم تحميل {len(data)} من 4 ملفات فقط")

    stats_detail_df, shapiro_df = build_human_statistical_results(data)

    if stats_detail_df is None or stats_detail_df.empty:
        st.warning("تعذّر حساب Mann-Whitney — تأكد من وجود الملفات الأربعة.")
        return

    if shapiro_df is not None and not shapiro_df.empty:
        st.subheader("Shapiro-Wilk")
        st.dataframe(shapiro_df, use_container_width=True, hide_index=True)

    st.subheader("Mann-Whitney U — LLaMA")
    st.dataframe(
        mann_whitney_table(stats_detail_df[stats_detail_df["النموذج"] == "LLaMA"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Mann-Whitney U — Qwen")
    st.dataframe(
        mann_whitney_table(stats_detail_df[stats_detail_df["النموذج"] == "Qwen"]),
        use_container_width=True,
        hide_index=True,
    )


main()
