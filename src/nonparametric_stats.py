"""Mann-Whitney U، Shapiro-Wilk، rank-biserial، وتصحيح Holm-Bonferroni."""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from scipy.stats import mannwhitneyu, shapiro

MIN_SAMPLES_DEFAULT = 3
MAX_SHAPIRO_N = 5000


def holm_bonferroni(p_values: Sequence[float]) -> np.ndarray:
    """تصحيح p-values متعددة الاختبارات بطريقة Holm-Bonferroni."""
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    adjusted_sorted = np.empty(len(p))
    for i, idx in enumerate(order):
        adjusted_sorted[i] = (len(p) - i) * p[idx]
    for i in range(1, len(adjusted_sorted)):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)
    adjusted = np.empty(len(p))
    adjusted[order] = adjusted_sorted
    return adjusted


def _clean_values(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)]


def shapiro_wilk(
    values: Sequence[float],
    *,
    min_n: int = MIN_SAMPLES_DEFAULT,
    max_n: int = MAX_SHAPIRO_N,
) -> dict[str, Any]:
    """Shapiro-Wilk — يُرجع n و W و p (أو None إذا حجم العينة خارج النطاق)."""
    arr = _clean_values(values)
    n = len(arr)
    result: dict[str, Any] = {"n": n, "W": None, "p": None}
    if n < min_n or n > max_n:
        return result
    stat, p_value = shapiro(arr)
    result["W"] = float(stat)
    result["p"] = float(p_value)
    return result


def shapiro_normality_label(p: Optional[float], *, alpha: float = 0.05) -> str:
    if p is None:
        return ""
    return "غير طبيعي" if p < alpha else "طبيعي تقريبًا"


def format_p_value(p: Optional[float]) -> str:
    """عرض p-value بالصيغة العلمية حتى لا تظهر كصفر في الجداول."""
    if p is None:
        return ""
    return format(float(p), ".10e")
#_________________________________________________

def apply_holm_to_rows(
    rows: list[dict],
    *,
    p_raw_key: str = "p_raw",
    holm_key: str = "p_holm",
    decision_key: Optional[str] = None,
    significant_label: str = "دال",
    not_significant_label: str = "غير دال",
    alpha: float = 0.05,
) -> None:
#_________________تطبق تصحيح هولم على قيم p الخام الموجودة في مجموعة من النتائج 

    """تطبيق Holm على صفوف تحتوي p_raw (تعديل in-place)."""
    eligible = [r for r in rows if r.get(p_raw_key) is not None]
    if not eligible:
        return
    adjusted = holm_bonferroni([float(r[p_raw_key]) for r in eligible])
    for row, p_adj in zip(eligible, adjusted):
        row[holm_key] = round(float(p_adj), 4)
        if decision_key:
            row[decision_key] = (
                significant_label if p_adj < alpha else not_significant_label
            )

#________________________________________________

def rank_biserial(u_stat: float, n_a: int, n_b: int) -> float:
    """معامل rank-biserial من إحصائية U (اختبار Mann-Whitney)."""
    return 1.0 - (2.0 * u_stat) / (n_a * n_b)


def effect_magnitude(r_rb: float) -> str:
    """تصنيف حجم الأثر لـ rank-biserial (عتبات Cohen لـ r: 0.10 / 0.30 / 0.50)."""
    abs_rb = abs(r_rb)
    if abs_rb < 0.10:
        return "ضعيف جداً"
    if abs_rb < 0.30:
        return "ضعيف"
    if abs_rb < 0.50:
        return "متوسط"
    return "قوي"


def mann_whitney_u(
    a: Sequence[float],
    b: Sequence[float],
    *,
    min_n: int = MIN_SAMPLES_DEFAULT,
) -> Optional[dict[str, Any]]:
    """
    Mann-Whitney U (ذو ذيلين).

    يُرجع None إذا كانت إحدى العينتين أصغر من min_n.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    a_arr = a_arr[~np.isnan(a_arr)]
    b_arr = b_arr[~np.isnan(b_arr)]
    if len(a_arr) < min_n or len(b_arr) < min_n:
        return None

    u_stat, p_value = mannwhitneyu(a_arr, b_arr, alternative="two-sided")
    rb = rank_biserial(float(u_stat), len(a_arr), len(b_arr))
    return {
        "u": float(u_stat),
        "p": float(p_value),
        "rb": rb,
        "n_a": len(a_arr),
        "n_b": len(b_arr),
        "median_a": float(np.median(a_arr)),
        "median_b": float(np.median(b_arr)),
        "mean_a": float(np.mean(a_arr)),
        "mean_b": float(np.mean(b_arr)),
    }


def mann_whitney_summary(
    a: Sequence[float],
    b: Sequence[float],
    *,
    min_n: int = MIN_SAMPLES_DEFAULT,
) -> Optional[dict[str, Any]]:
    """نتيجة Mann-Whitney جاهزة للعرض (قيم مُقرّبة + حجم الأثر)."""
    raw = mann_whitney_u(a, b, min_n=min_n)
    if raw is None:
        return None
    return {
        "u": round(raw["u"], 2),
        "p": round(raw["p"], 4),
        "rb": round(raw["rb"], 4),
        "effect": effect_magnitude(raw["rb"]),
        "med_a": round(raw["median_a"], 4),
        "med_b": round(raw["median_b"], 4),
        "mean_a": round(raw["mean_a"], 4),
        "mean_b": round(raw["mean_b"], 4),
        "n_a": raw["n_a"],
        "n_b": raw["n_b"],
    }
