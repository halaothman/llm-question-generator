"""
RAG_VERSION config diagram: baseline vs improved branches (colored).

Output:
  OUTPUT/rag_version_config.png
  docs/figures/rag_version_config.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT_PATHS = [
    ROOT / "OUTPUT" / "rag_version_config.png",
    ROOT / "docs" / "figures" / "rag_version_config.png",
]

C = {
    "text": "#1F2937",
    "arrow": "#374151",
    "root_fill": "#EDE9FE",
    "root_edge": "#7C3AED",
    "baseline_fill": "#DBEAFE",
    "baseline_edge": "#2563EB",
    "baseline_db": "#BFDBFE",
    "improved_fill": "#D1FAE5",
    "improved_edge": "#059669",
    "improved_db": "#A7F3D0",
    "label_fill": "#F3F4F6",
    "label_edge": "#9CA3AF",
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


def _rounded_box(ax, x, y, w, h, text, *, fill, edge, fontsize=9.5, title_size=10.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.08",
            linewidth=1.4,
            edgecolor=edge,
            facecolor=fill,
            zorder=2,
        )
    )
    lines = text.split("\n")
    if len(lines) == 1:
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
        )
        return
    ax.text(
        x + w / 2,
        y + h - 0.28,
        lines[0],
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=C["text"],
        zorder=3,
    )
    ax.text(
        x + w / 2,
        y + h / 2 - 0.08,
        "\n".join(lines[1:]),
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=C["text"],
        zorder=3,
        linespacing=1.25,
    )


def _db_cylinder(ax, x, y, w, h, text, *, fill, edge):
    body_h = h * 0.72
    rim_h = h * 0.14
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            body_h,
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.4,
            zorder=2,
        )
    )
    ax.add_patch(
        Arc(
            (x + w / 2, y + body_h),
            w,
            rim_h * 2,
            angle=0,
            theta1=0,
            theta2=180,
            edgecolor=edge,
            facecolor=fill,
            linewidth=1.4,
            zorder=3,
        )
    )
    ax.add_patch(
        Arc(
            (x + w / 2, y),
            w,
            rim_h * 2,
            angle=0,
            theta1=180,
            theta2=360,
            edgecolor=edge,
            facecolor=fill,
            linewidth=1.4,
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        y + body_h / 2,
        text,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=C["text"],
        zorder=4,
    )


def _arrow(ax, x1, y1, x2, y2, *, rad=0.0, color=None):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=color or C["arrow"],
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
    )


def _edge_label(ax, x, y, text):
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.34, y - 0.11),
            0.68,
            0.22,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            linewidth=0.9,
            edgecolor=C["label_edge"],
            facecolor=C["label_fill"],
            zorder=4,
        )
    )
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=C["text"],
        zorder=5,
    )


def build_figure() -> plt.Figure:
    fig_w, fig_h = 8.4, 5.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.set_facecolor("white")

    root_w, root_h = 2.2, 0.95
    root_x = (fig_w - root_w) / 2
    root_y = 4.55
    _rounded_box(
        ax,
        root_x,
        root_y,
        root_w,
        root_h,
        "config.py\nRAG_VERSION",
        fill=C["root_fill"],
        edge=C["root_edge"],
        fontsize=9.0,
        title_size=10.0,
    )

    box_w, box_h = 2.55, 1.05
    left_x, right_x = 0.75, fig_w - 0.75 - box_w
    mid_y = 2.45

    _rounded_box(
        ax,
        left_x,
        mid_y,
        box_w,
        box_h,
        "baseline_model/\ne5 · top_k=5 · thr=0.82",
        fill=C["baseline_fill"],
        edge=C["baseline_edge"],
    )
    _rounded_box(
        ax,
        right_x,
        mid_y,
        box_w,
        box_h + 0.32,
        "improved_model/\nAraBERT · thr=0.65\nFAISS top-10 → rerank → top-5",
        fill=C["improved_fill"],
        edge=C["improved_edge"],
        fontsize=8.8,
    )

    db_w, db_h = 2.05, 0.82
    db_y = 0.55
    _db_cylinder(
        ax,
        left_x + (box_w - db_w) / 2,
        db_y,
        db_w,
        db_h,
        "indexes/baseline.*",
        fill=C["baseline_db"],
        edge=C["baseline_edge"],
    )
    _db_cylinder(
        ax,
        right_x + (box_w - db_w) / 2,
        db_y,
        db_w,
        db_h,
        "indexes/improved.*",
        fill=C["improved_db"],
        edge=C["improved_edge"],
    )

    root_cx = root_x + root_w / 2
    root_bottom = root_y
    left_cx = left_x + box_w / 2
    right_cx = right_x + box_w / 2
    mid_top = mid_y + box_h + 0.32
    left_top = mid_y + box_h
    db_top = db_y + db_h * 0.86

    _arrow(ax, root_cx - 0.55, root_bottom - 0.02, left_cx, mid_top + 0.02, rad=0.22)
    _arrow(ax, root_cx + 0.55, root_bottom - 0.02, right_cx, mid_top + 0.2, rad=-0.22)
    _edge_label(ax, root_cx - 1.05, 3.95, "baseline")
    _edge_label(ax, root_cx + 1.05, 3.95, "improved")

    _arrow(ax, left_cx, mid_y - 0.02, left_cx, db_top + 0.02, rad=0.0)
    _arrow(ax, right_cx, mid_y - 0.02, right_cx, db_top + 0.02, rad=0.0)

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
