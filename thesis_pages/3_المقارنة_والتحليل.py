"""صفحة المقارنة والتحليل: Mann-Whitney وShapiro ومخططات Vanilla مقابل RAG."""
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from src.nonparametric_stats import effect_magnitude, format_p_value, mann_whitney_u, shapiro_wilk
from src.question_dataset import (
    METRICS,
    OUTPUTS_DIR,
    load_question_dataframe,
)
from src.ui_styles import inject_app_styles

st.set_page_config(page_title="المقارنة والتحليل", layout="wide")

inject_app_styles()

# مجلد حفظ صور المخططات المصدّرة من زر «تصدير Violin»
PLOTS_DIR = Path("plots")
# المقاييس المتاحة في القائمة المنسدلة (log_perplexity مشتق من perplexity)
METRIC_OPTIONS = METRICS + ["log_perplexity"]
# مقاييس يُفترض أن القيمة الأعلى فيها أفضل (لضبط محور Y في المخطط)
HIGHER_IS_BETTER = {"precision", "recall", "f1_score", "bleu", "bert_score"}
# ألوان ثابتة لتمييز Vanilla عن RAG في المخططات
COLORS = {"Vanilla": "#1f77b4", "RAG": "#ff7f0e"}


def rtl_md(text: str) -> None:
    """عرض نص Markdown في الصفحة (عناوين، فواصل، إلخ)."""
    st.markdown(text.strip(), unsafe_allow_html=True)


def embed_plotly(fig, height_px: int) -> None:
    """تضمين مخطط Plotly داخل iframe مع شريط أدواته الافتراضي (تكبير، تصغير، PNG)."""
    fig.update_layout(
        height=height_px,
        autosize=False,
        margin=dict(l=52, r=44, t=76, b=56),
    )
    html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": True, "responsive": True},
    )
    html = html.replace("<html", '<html dir="ltr"', 1)
    components.html(html, height=height_px + 72, scrolling=True)


def filter_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """تصفية الصفوف الصالحة للمقياس المختار (استبعاد NaN وقيم perplexity الفاشلة)."""
    if metric in ("perplexity", "log_perplexity"):
        out = df[
            df["perplexity"].notna()
            & (df["perplexity"] != 0.0)
            & (df["perplexity"] != 1000.0)
        ]
        if metric == "log_perplexity":
            out = out[out["log_perplexity"].notna()]
        return out
    return df.dropna(subset=[metric])


def _significance_note(p: float, *, alpha: float = 0.05) -> str:
    """ترجمة p-value إلى ملاحظة عربية عن وجود فرق إحصائي."""
    return "فرق ذو دلالة" if p < alpha else "لا فرق ذو دلالة"


def shapiro_table(df: pd.DataFrame, metric: str) -> None:
    """جدول Shapiro-Wilk: اختبار التوزيع الطبيعي لكل (نموذج × طريقة)."""
    sub = filter_metric(df, metric)
    rows = []
    for (model, method), g in sub.groupby(["model", "method"]):
        vals = g[metric].to_numpy()
        sw = shapiro_wilk(vals)
        rows.append(
            {
                "النموذج": model,
                "الطريقة": method,
                "حجم العينة": sw["n"],
                "W": sw["W"],
                "p": format_p_value(sw["p"]),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def mann_whitney_vanilla_rag(df: pd.DataFrame, metric: str, version: Optional[str]) -> None:
    """جدول Mann-Whitney: مقارنة Vanilla مقابل RAG لكل نموذج (Baseline أو Improved)."""
    label = {"before": "Baseline", "after": "Improved"}.get(version or "", "الكل")
    rtl_md(f"### Mann-Whitney: Vanilla vs RAG ({label})")
    sub = filter_metric(df, metric)
    if version:
        sub = sub[sub["version"] == version]
    rows = []
    for model, g in sub.groupby("model"):
        v = g[g["method"] == "Vanilla"][metric].to_numpy()
        r = g[g["method"] == "RAG"][metric].to_numpy()
        res = mann_whitney_u(v, r)
        if not res:
            rows.append({"النموذج": model, "n Vanilla": len(v), "n RAG": len(r)})
            continue
        rows.append(
            {
                "النموذج": model,
                "n Vanilla": res["n_a"],
                "n RAG": res["n_b"],
                "U": round(res["u"], 2),
                "p": round(res["p"], 4),
                "Rank-Biserial": round(res["rb"], 4),
                "حجم الأثر": effect_magnitude(res["rb"]),
                "ملاحظة": _significance_note(res["p"]),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def y_range(metric: str, values: pd.Series, adaptive: bool):
    """حدود محور Y المناسبة لكل مقياس (0–1 للجودة، أو نطاق تكيّفي لـ perplexity)."""
    if metric in HIGHER_IS_BETTER:
        if adaptive and len(values) and (values >= 0.9).mean() > 0.9:
            return [max(0, float(values.min()) - 0.05), min(1.0, float(values.max()) + 0.05)]
        return [0, 1]
    if metric == "perplexity":
        return [max(0, float(values.min()) - 10), float(values.quantile(0.95)) + 10]
    if metric == "log_perplexity":
        return [float(values.min()) - 0.5, float(values.max()) + 0.5]
    return None


def export_plot(fig, filename: str) -> Path:
    """حفظ المخطط كصورة PNG عالية الدقة داخل مجلد plots/."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    fig.write_image(str(path), width=1200, height=800, scale=2)
    return path


def plot_violin(plot_df: pd.DataFrame, metric: str, facet_row: Optional[str], prefix: str) -> None:
    """رسم مخطط Violin (Vanilla vs RAG) مع تقسيم حسب النموذج ونسخة RAG، وزر تصدير PNG."""
    yr = y_range(metric, plot_df[metric], True)
    kwargs = dict(
        x="method",
        y=metric,
        color="method",
        facet_col="model",
        hover_data=["file", "source"],
        color_discrete_map=COLORS,
    )
    if facet_row:
        kwargs["facet_row"] = facet_row
        kwargs["category_orders"] = {"version": ["before", "after"]}

    fig = px.violin(plot_df, box=True, points="outliers", **kwargs)
    fig.update_layout(title=f"Violin — {metric}", template="plotly_white", legend_title="الطريقة")
    fig.update_yaxes(range=yr, tickformat=".4f")
    if facet_row:
        fig.for_each_annotation(
            lambda a: a.update(
                text=a.text.replace("version=before", "Baseline").replace("version=after", "Improved")
            )
        )
    embed_plotly(fig, 940 if facet_row else 560)
    if st.button("تصدير Violin", key=f"{prefix}_violin_{metric}"):
        try:
            st.success(f"تم الحفظ: `{export_plot(fig, f'{prefix}_violin_{metric}.png')}`")
        except Exception as e:
            st.error(str(e))


def main():
    """نقطة الدخول: تحميل البيانات، الجداول الإحصائية، ومخطط Violin للمقياس المختار."""
    rtl_md("# المقارنة والتحليل")

    question_df = load_question_dataframe(OUTPUTS_DIR)
    if question_df.empty:
        st.warning("لا توجد ملفات أسئلة في `outputs/`.")
        return

    metric = st.selectbox("المقياس", METRIC_OPTIONS, index=0)

    rtl_md("## Shapiro-Wilk")
    shapiro_table(question_df, metric)

    rtl_md("---")
    rtl_md("## أثر RAG")
    mann_whitney_vanilla_rag(question_df, metric, "before")
    mann_whitney_vanilla_rag(question_df, metric, "after")

    rtl_md("---")
    rtl_md("## المخططات")
    plot_df = filter_metric(question_df, metric)
    if plot_df.empty:
        st.info("لا توجد بيانات للرسم.")
        return
    plot_violin(plot_df, metric, facet_row="version", prefix="rag_effect")


main()
