"""
Six-stage reference indexing pipeline (2×3 grid) — matches thesis implementation.

Output:
  OUTPUT/data_preparation_pipeline.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "OUTPUT"

TITLE_COLOR = "#1F2937"
BODY_COLOR = "#374151"
BORDER_COLOR = "#4B5563"
ARROW_COLOR = "#4B5563"

STAGE_FILLS = [
    "#DBEAFE",
    "#D6EAF8",
    "#EBF5FB",
    "#E8F4FC",
    "#D5F5E3",
    "#EDE9FE",
]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Tahoma", "Arial"],
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.pad_inches": 0.10,
    }
)

STAGES = [
    {
        "title": "Reference documents",
        "lines": ["data-sources/", "PDF · DOCX · TXT · MD"],
        "bullets": False,
    },
    {
        "title": "Text Extraction",
        "lines": ["loaders.load_text()", "Plain text"],
        "bullets": True,
    },
    {
        "title": "Arabic Text\nPreprocessing",
        "lines": [
            "clean_ar: diacritics, punct.",
            "Whitespace collapse",
            "stem_ar",
        ],
        "bullets": True,
    },
    {
        "title": "Chunking",
        "lines": ["chunk_text()", "500 chars, 100 overlap"],
        "bullets": True,
    },
    {
        "title": "Embedding\n(index one model)",
        "lines": [
            "Baseline: e5-large-v2",
            "passage: prefix; L2 norm",
            "Improved: AraBERTv2; L2 norm",
        ],
        "bullets": True,
    },
    {
        "title": "FAISS + JSONL",
        "lines": [
            "IndexFlatIP",
            "JSONL: id, text",
            "filename, chunk_size",
        ],
        "bullets": True,
    },
]

LAYOUT = {
    "fig_w": 10.2,
    "fig_h": 5.6,
    "cols": 3,
    "rows": 2,
    "box_w": 2.85,
    "box_h": 2.18,
    "gap_x": 0.38,
    "gap_y": 0.48,
    "title_size": 10.8,
    "line_size": 8.5,
    "title_leading": 0.18,
    "body_leading": 0.155,
    "block_gap": 0.10,
}


def _box(ax, x, y, w, h, fill):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.05",
            linewidth=1.1,
            edgecolor=BORDER_COLOR,
            facecolor=fill,
            zorder=2,
        )
    )


def _arrow_h(ax, x1, x2, y):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y),
            (x2, y),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=ARROW_COLOR,
            shrinkA=2,
            shrinkB=2,
            zorder=1,
        )
    )


def _arrow_l(ax, x1, y1, x2, y2, y_mid):
    kw = dict(
        arrowstyle="-",
        mutation_scale=10,
        linewidth=1.0,
        color=ARROW_COLOR,
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    )
    ax.add_patch(FancyArrowPatch((x1, y1), (x1, y_mid), **kw))
    ax.add_patch(FancyArrowPatch((x1, y_mid), (x2, y_mid), **kw))
    ax.add_patch(
        FancyArrowPatch(
            (x2, y_mid),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=ARROW_COLOR,
            shrinkA=0,
            shrinkB=2,
            zorder=1,
        )
    )


def _stage_pos(cfg, index: int) -> tuple[float, float]:
    cols, rows = cfg["cols"], cfg["rows"]
    box_w, box_h = cfg["box_w"], cfg["box_h"]
    gap_x, gap_y = cfg["gap_x"], cfg["gap_y"]
    total_w = cols * box_w + (cols - 1) * gap_x
    total_h = rows * box_h + (rows - 1) * gap_y
    x0 = (cfg["fig_w"] - total_w) / 2
    y0 = (cfg["fig_h"] - total_h) / 2
    col = index % cols
    row = index // cols
    x = x0 + col * (box_w + gap_x)
    y = y0 + (rows - 1 - row) * (box_h + gap_y)
    return x, y


def _draw_stage(ax, x, y, w, h, stage, fill, cfg):
    _box(ax, x, y, w, h, fill)
    cx = x + w / 2
    cy = y + h / 2
    title_lines = stage["title"].count("\n") + 1
    body_lines = len(stage["lines"])
    title_block = title_lines * cfg["title_leading"]
    body_block = body_lines * cfg["body_leading"]
    block_gap = cfg["block_gap"]
    total_h = title_block + block_gap + body_block
    y_top = cy + total_h / 2

    ax.text(
        cx,
        y_top,
        stage["title"],
        ha="center",
        va="top",
        fontsize=cfg["title_size"],
        fontweight="bold",
        color=TITLE_COLOR,
        zorder=3,
        linespacing=1.08,
        clip_on=False,
    )

    bullet = "• " if stage["bullets"] else ""
    body = "\n".join(f"{bullet}{line}" for line in stage["lines"])
    ax.text(
        cx,
        y_top - title_block - block_gap,
        body,
        ha="center",
        va="top",
        fontsize=cfg["line_size"],
        fontweight="bold",
        color=BODY_COLOR,
        zorder=3,
        linespacing=1.18,
        clip_on=False,
    )


def build_data_preparation_grid_figure() -> plt.Figure:
    cfg = LAYOUT
    box_w, box_h = cfg["box_w"], cfg["box_h"]
    gap_x, gap_y = cfg["gap_x"], cfg["gap_y"]
    fig_w, fig_h = cfg["fig_w"], cfg["fig_h"]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.set_facecolor("white")

    positions = []
    for i, stage in enumerate(STAGES):
        x, y = _stage_pos(cfg, i)
        positions.append((x, y))
        _draw_stage(ax, x, y, box_w, box_h, stage, STAGE_FILLS[i], cfg)

    for i in range(2):
        x1, y1 = positions[i]
        x2, _y2 = positions[i + 1]
        cy = y1 + box_h / 2
        _arrow_h(ax, x1 + box_w + 0.03, x2 - 0.03, cy)

    for i in range(3, 5):
        x1, y1 = positions[i]
        x2, _y2 = positions[i + 1]
        cy = y1 + box_h / 2
        _arrow_h(ax, x1 + box_w + 0.03, x2 - 0.03, cy)

    x3, y3 = positions[2]
    x4, y4 = positions[3]
    y_mid = y4 + box_h + gap_y / 2
    _arrow_l(
        ax,
        x3 + box_w / 2,
        y3 - 0.03,
        x4 + box_w / 2,
        y4 + box_h + 0.03,
        y_mid,
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.04)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = build_data_preparation_grid_figure()
    path = OUT / "data_preparation_pipeline.png"
    fig.savefig(path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
