"""Generate the classic five-phase thesis methodology diagram to OUTPUT/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "OUTPUT"

C = {
    "phase": "#F5F5F5",
    "phase_edge": "#CCCCCC",
    "input": "#D6EAF8",
    "process": "#EBF5FB",
    "retrieval": "#FCF3CF",
    "retrieval_improved": "#F9E79F",
    "generation": "#D5F5E3",
    "dataset": "#E8F8F5",
    "evaluation": "#E8DAEF",
    "stats": "#FADBD8",
    "border": "#1F2937",
    "arrow": "#374151",
    "arrow_light": "#6B7280",
    "text": "#1F2937",
    "muted": "#374151",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Tahoma", "DejaVu Sans", "Arial"],
        "font.size": 9.0,
        "font.weight": "bold",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.pad_inches": 0.08,
    }
)


def _box(ax, x, y, w, h, text, facecolor, fontsize=8.5, edgecolor=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.06",
            linewidth=1.25,
            edgecolor=edgecolor or C["border"],
            facecolor=facecolor,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=C["text"],
        zorder=3,
        linespacing=1.22,
    )


def _arrow(ax, x1, y1, x2, y2, color=None, connection="arc3,rad=0", lw=1.1):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            color=color or C["arrow"],
            linewidth=lw,
            mutation_scale=11,
            connectionstyle=connection,
            zorder=1,
        )
    )


def _phase_band(ax, y, h, label):
    ax.add_patch(
        Rectangle(
            (0.05, y),
            0.14,
            h,
            facecolor=C["phase"],
            edgecolor=C["phase_edge"],
            linewidth=0.8,
            zorder=0,
        )
    )
    ax.text(
        0.12,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=C["text"],
        rotation=90,
        zorder=1,
    )


def build_methodology_figure() -> plt.Figure:
    fig_w, fig_h = 10.5, 13.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 13.6)
    ax.axis("off")

    _phase_band(ax, 11.2, 1.05, "Phase I\nData")
    _box(ax, 0.55, 12.15, 2.0, 0.72, "Arabic educational\ndocuments", C["input"], fontsize=8.8)
    _box(ax, 2.75, 12.15, 1.75, 0.72, "Extract &\nnormalize", C["process"], fontsize=8.8)
    _box(ax, 4.7, 12.15, 1.65, 0.72, "Text\nchunking", C["process"], fontsize=8.8)
    _box(
        ax,
        6.55,
        12.15,
        3.4,
        0.72,
        "FAISS index \u00b7 embeddings \u00b7 metadata",
        C["process"],
        fontsize=8.5,
    )
    for x0, x1 in [(2.55, 2.75), (4.5, 4.7), (6.35, 6.55)]:
        _arrow(ax, x0, 12.51, x1, 12.51)

    _phase_band(ax, 8.55, 2.15, "Phase II\nRetrieval")
    _box(
        ax,
        0.55,
        10.75,
        9.4,
        0.72,
        "Retrieval pipeline:  query embedding  \u2192  FAISS search  \u2192  Top-K retrieval  \u2192  "
        "retrieved context",
        C["retrieval"],
        fontsize=8.3,
    )
    _arrow(ax, 8.25, 12.15, 5.25, 11.47, connection="arc3,rad=-0.08")
    _arrow(ax, 5.25, 10.75, 2.725, 10.18)
    _arrow(ax, 5.25, 10.75, 7.75, 10.18)
    _box(
        ax,
        0.55,
        9.25,
        4.35,
        0.88,
        "Baseline retrieval configuration\nE5 \u00b7 top-5 \u00b7 no re-ranking \u00b7 \u03b8 \u2265 0.82",
        C["retrieval"],
        fontsize=8.1,
        edgecolor="#B7950B",
    )
    _box(
        ax,
        5.55,
        9.25,
        4.4,
        0.88,
        "Improved retrieval configuration\nAraBERT \u00b7 top-10 \u2192 rerank \u00b7 \u03b8 \u2265 0.65",
        C["retrieval_improved"],
        fontsize=8.1,
        edgecolor="#27AE60",
    )

    _phase_band(ax, 4.35, 3.65, "Phase III\nGeneration")
    _box(ax, 3.55, 8.35, 3.4, 0.52, "Input document", C["input"], fontsize=9.0)
    _box(ax, 0.55, 7.15, 3.2, 0.72, "Vanilla prompt", C["generation"], fontsize=8.8)
    _arrow(ax, 4.0, 8.35, 2.15, 7.87)
    _box(ax, 6.75, 7.65, 3.2, 0.62, "Context retrieval", C["generation"], fontsize=8.5)
    _box(ax, 6.75, 6.75, 3.2, 0.62, "Retrieved context", C["generation"], fontsize=8.5)
    _box(ax, 6.75, 5.85, 3.2, 0.72, "RAG prompt", C["generation"], fontsize=8.8)
    _arrow(ax, 6.0, 8.35, 8.35, 8.27)
    _arrow(ax, 5.25, 10.75, 8.35, 8.27, color=C["arrow_light"], lw=0.9)
    _arrow(ax, 8.35, 7.65, 8.35, 7.37)
    _arrow(ax, 8.35, 6.75, 8.35, 6.47)
    _box(
        ax,
        2.65,
        4.0,
        5.2,
        1.0,
        "Ollama Inference\nLLaMA 3.2 3B\nQwen 2.5 7B\nTemperature = 0.7",
        C["generation"],
        fontsize=8.3,
    )
    _arrow(ax, 2.15, 7.15, 4.5, 5.0)
    _arrow(ax, 8.35, 5.85, 6.85, 5.0)
    _box(
        ax,
        0.55,
        3.2,
        9.4,
        0.58,
        "Validated JSON \u00b7 MCQ \u00b7 True/False \u00b7 metadata (model, method, source, file id)",
        C["generation"],
        fontsize=8.3,
    )
    _box(
        ax,
        0.55,
        2.5,
        9.4,
        0.52,
        "Validated JSON \u2192 Post-processing (Validation & Duplicate Removal) \u2192 Save to outputs/",
        C["dataset"],
        fontsize=7.85,
    )
    _arrow(ax, 5.25, 4.0, 5.25, 3.78)
    _arrow(ax, 5.25, 3.2, 5.25, 3.02)

    _phase_band(ax, 1.15, 2.35, "Phase IV\nEvaluation")
    _box(
        ax,
        0.55,
        1.10,
        4.35,
        1.15,
        "Automatic evaluation\nPrecision \u00b7 Recall \u00b7 F1 \u00b7 BLEU \u00b7 BERTScore \u00b7 Perplexity",
        C["evaluation"],
        fontsize=8.3,
    )
    _box(
        ax,
        5.55,
        1.10,
        4.4,
        1.15,
        "Human evaluation\nSampled questions \u00b7 single-blind \u00b7 Likert 1\u20135\n"
        "Clarity \u00b7 Logic \u00b7 Relevance \u00b7 Option quality \u00b7 Accuracy",
        C["evaluation"],
        fontsize=8.1,
    )
    _arrow(ax, 2.72, 2.5, 2.72, 2.25)
    _arrow(ax, 7.75, 2.5, 7.75, 2.25)

    _phase_band(ax, 0.15, 1.05, "Phase V\nStatistics")
    _box(ax, 0.55, 0.35, 2.05, 0.72, "Shapiro\u2013Wilk", C["stats"], fontsize=8.3)
    _box(ax, 2.85, 0.35, 2.15, 0.72, "Mann\u2013Whitney U", C["stats"], fontsize=8.3)
    _box(ax, 5.25, 0.35, 2.15, 0.72, "Holm\u2013Bonferroni", C["stats"], fontsize=8.3)
    _box(ax, 7.65, 0.35, 2.3, 0.72, "Effect size (r_rb)", C["stats"], fontsize=8.3)
    for x0, x1 in [(2.6, 2.85), (5.0, 5.25), (7.4, 7.65)]:
        _arrow(ax, x0, 0.71, x1, 0.71)
    _arrow(ax, 2.72, 1.10, 1.57, 1.07)
    _arrow(ax, 7.75, 1.10, 6.32, 1.07)

    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = build_methodology_figure()
    for name in ("thesis_methodology_five_phases.png", "thesis_methodology_implemented.png"):
        path = OUT / name
        fig.savefig(path, facecolor="white", edgecolor="none")
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
