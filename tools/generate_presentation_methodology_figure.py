"""
Compact thesis methodology — clear vertical flow, no crossing arrows.

Output:
  OUTPUT/presentation_methodology.png
  docs/figures/presentation_methodology.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT_PATHS = [
    ROOT / "OUTPUT" / "presentation_methodology.png",
    ROOT / "docs" / "figures" / "presentation_methodology.png",
]

C = {
    "text": "#1F2937",
    "arrow": "#334155",
    "section": "#F1F5F9",
    "section_edge": "#CBD5E1",
    "sources": ("#DBEAFE", "#2563EB"),
    "index": ("#D6EAF8", "#1D4ED8"),
    "upload": ("#E0E7FF", "#4338CA"),
    "vanilla": ("#FFEDD5", "#EA580C"),
    "rag": ("#FEF9C3", "#CA8A04"),
    "gen": ("#D5F5E3", "#16A34A"),
    "ollama": ("#DCFCE7", "#15803D"),
    "out": ("#E0F2FE", "#0284C7"),
    "auto": ("#EDE9FE", "#7C3AED"),
    "human": ("#FCE7F3", "#DB2777"),
    "stats": ("#FEE2E2", "#DC2626"),
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Tahoma", "Arial"],
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.pad_inches": 0.08,
    }
)

FIG_W = 5.4
CX = FIG_W / 2
BOX_W = 2.55
LANE_W = 2.15
BOX_H = 0.42
GAP = 0.18
LEFT_CX = 1.35
RIGHT_CX = FIG_W - 1.35


def _box(ax, cx, y, w, h, text, colors, *, fontsize=8.8, sub_size=7.2):
    fill, edge = colors
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, y), w, h,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            linewidth=1.25, edgecolor=edge, facecolor=fill, zorder=2,
        )
    )
    lines = text.split("\n")
    if len(lines) == 1:
        ax.text(cx, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=C["text"], zorder=3)
        return
    ax.text(cx, y + h - 0.09, lines[0], ha="center", va="top",
            fontsize=fontsize, fontweight="bold", color=C["text"], zorder=3)
    ax.text(cx, y + (h - 0.14) / 2, "\n".join(lines[1:]), ha="center", va="center",
            fontsize=sub_size, fontweight="bold", color=C["text"], zorder=3, linespacing=1.08)


def _arrow(ax, x1, y1, x2, y2, *, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=11, linewidth=1.05,
            color=C["arrow"], connectionstyle=f"arc3,rad={rad}",
            shrinkA=3, shrinkB=3, zorder=1,
        )
    )


def _section(ax, y, h):
    ax.add_patch(
        FancyBboxPatch(
            (0.12, y), FIG_W - 0.24, h,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            linewidth=0.9, edgecolor=C["section_edge"], facecolor=C["section"], zorder=0,
        )
    )


def _divider(ax, y):
    ax.plot([0.35, FIG_W - 0.35], [y, y], color=C["section_edge"], lw=1.0, zorder=1)


def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(FIG_W, 7.2))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_facecolor("white")

    # ── Phase 1: knowledge base (RAG prep only) ──
    _section(ax, 5.55, 1.55)
    y = 6.55
    _box(ax, CX, y, BOX_W, BOX_H, "data-sources/  ·  PDF · DOCX · TXT", C["sources"], fontsize=8.4)
    y -= BOX_H + GAP
    _arrow(ax, CX, y + BOX_H + GAP, CX, y + BOX_H + 0.02)
    _box(ax, CX, y, BOX_W, BOX_H + 0.04, "Indexing  ·  chunk · embed · FAISS", C["index"], fontsize=8.2)

    _divider(ax, 5.42)

    # ── Phase 2: generation ──
    _section(ax, 2.05, 3.22)
    y = 4.85
    _box(ax, CX, y, BOX_W, BOX_H, "Upload file  ·  uploads/", C["upload"], fontsize=8.4)
    y -= BOX_H + GAP + 0.04
    _arrow(ax, CX, y + BOX_H + GAP + 0.04, LEFT_CX, y + 0.62)
    _arrow(ax, CX, y + BOX_H + GAP + 0.04, RIGHT_CX, y + 0.62)

    fork_h = 0.58
    _box(ax, LEFT_CX, y, LANE_W, fork_h, "Vanilla\nupload text · no retrieval", C["vanilla"], sub_size=6.8)
    _box(
        ax, RIGHT_CX, y, LANE_W, fork_h,
        "RAG\nBaseline OR Improved · retrieve from FAISS",
        C["rag"], fontsize=8.0, sub_size=6.6,
    )

    y -= fork_h + GAP
    _arrow(ax, LEFT_CX, y + fork_h + GAP, CX - 0.35, y + BOX_H + 0.02, rad=0.08)
    _arrow(ax, RIGHT_CX, y + fork_h + GAP, CX + 0.35, y + BOX_H + 0.02, rad=-0.08)
    _box(ax, CX, y, BOX_W, BOX_H, "Prompt  ·  Vanilla or RAG", C["gen"], fontsize=8.4)

    y -= BOX_H + GAP
    _arrow(ax, CX, y + BOX_H + GAP, CX, y + BOX_H + 0.02)
    _box(ax, CX, y, BOX_W, BOX_H, "Ollama  ·  LLaMA 3.2 & Qwen 2.5  ·  T=0.7", C["ollama"], fontsize=7.6)

    y -= BOX_H + GAP
    _arrow(ax, CX, y + BOX_H + GAP, CX, y + BOX_H + 0.02)
    _box(ax, CX, y, BOX_W, BOX_H, "JSON  →  outputs/", C["out"], fontsize=8.4)

    _divider(ax, 1.92)

    # ── Phase 3: evaluation ──
    _section(ax, 0.18, 1.68)
    eval_h = 0.46
    eval_y = 1.05
    _arrow(ax, CX, y, CX - 0.75, eval_y + eval_h)
    _arrow(ax, CX, y, CX + 0.75, eval_y + eval_h)
    _box(ax, CX - 0.82, eval_y, LANE_W - 0.15, eval_h, "Auto metrics", C["auto"], fontsize=8.2)
    _box(ax, CX + 0.82, eval_y, LANE_W - 0.15, eval_h, "Human eval", C["human"], fontsize=8.2)

    stats_y = 0.28
    _arrow(ax, CX - 0.82, eval_y, CX - 0.28, stats_y + BOX_H, rad=0.08)
    _arrow(ax, CX + 0.82, eval_y, CX + 0.28, stats_y + BOX_H, rad=-0.08)
    _box(ax, CX, stats_y, BOX_W, BOX_H, "Nonparametric stats", C["stats"], fontsize=8.4)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    return fig


def main() -> None:
    fig = build_figure()
    for path in OUT_PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor="white", edgecolor="none")
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
