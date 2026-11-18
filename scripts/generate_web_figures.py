#!/usr/bin/env python3
"""Redraw the project-page figure in the RADIAITE palette, and check the
published numbers agree wherever they are repeated.

This is deliberately separate from ``generate_figures.py``:

* ``generate_figures.py`` builds the figures for the paper from
  ``results/*.json``, and its output is what arXiv hosts. It is not touched
  here — the published figures stay exactly as published.
* This script builds ``docs/assets/recall_at_k.png`` for the project page, so
  the chart sits in the same visual system as the rest of the page.

It reads no experiment output. ``paper/main.tex`` is the authoritative record,
being what arXiv published, and every number here is transcribed from it. That
keeps the script runnable from a fresh clone with no Azure credentials and no
results.

The same figures get quoted in five places — the manuscript, README.md §4.1,
the project page's results table, the leaderboard at the top of that page
(twice per row: as text and as a bar width), and the SERIES constant below —
and they have drifted before. ``check()`` re-reads the manuscript and asserts
every other copy still matches it, plus that each <img> still states the size
of the file it points at. It needs nothing but the standard library, so CI runs
it on every push.

Usage:
    python scripts/generate_web_figures.py --check      # verify only, no draw
    python scripts/generate_web_figures.py [--font-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import re
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "docs", "assets")

# ── RADIAITE palette ──────────────────────────────────────────────
# Hue is the retrieval family, value is the rank within it — matching the
# four-way legend the page prints above this chart, so query-side stays in the
# page's quiet taupe rather than taking a colour the legend assigns to nothing.
#
# Seven of these must equal a token in the :root block of docs/index.html:
# PAPER_000, the four INK_*, FUSION[2] (amber-300) and QUERYSIDE[2]
# (paper-400). The rest are intermediate shades the chart needs and the page
# has no use for. Nothing enforces the seven — unlike the numbers below, a
# chart one shade off is invisible rather than wrong.
PAPER_000 = "#fcfaf4"
INK_900, INK_800, INK_600, INK_400 = "#17222e", "#22303f", "#3e4c5e", "#8091a3"
RULE_FAINT = "#e3e0d6"  # chart-only: the page draws its rules with color-mix()

FUSION = ["#866312", "#c08d2b", "#e4c878"]  # amber-700 / 500 / 300
LEXICAL = INK_800
SEMANTIC = ["#4a5631", "#aeb490"]  # moss-600 / 300
QUERYSIDE = ["#6f6650", "#9c9179", "#c5bca6"]  # paper-400 and two darker steps

DASH = {
    "fusion": "solid",
    "lexical": (0, (5, 2)),
    "semantic": (0, (5, 2, 1, 2)),
    "queryside": (0, (1, 2)),
}
# The chart carries the paper's two findings, so it gives them the weight.
LEAD_FAMILIES = ("fusion", "lexical")

K_VALUES = [1, 3, 5, 10, 20]

# (label, family, colour, marker, Recall@k for k in K_VALUES)
# Markers intentionally match MARKER in generate_figures.py, so a method means
# the same glyph in the paper and on the page. check() verifies the numbers.
# `None` is a value that does not exist: Hybrid + Rerank returns at most 10
# documents, so it has no Recall@20.
SERIES = [
    ("Hybrid + Rerank", "fusion", FUSION[0], "h", [0.472, 0.758, 0.816, 0.861, None]),
    ("Contextual Hybrid", "fusion", FUSION[1], "*", [0.327, 0.610, 0.717, 0.818, 0.887]),
    ("Hybrid RRF", "fusion", FUSION[2], "X", [0.308, 0.588, 0.695, 0.801, 0.877]),
    ("BM25", "lexical", LEXICAL, "o", [0.293, 0.552, 0.644, 0.735, 0.797]),
    ("CRAG", "queryside", QUERYSIDE[0], "^", [0.302, 0.556, 0.658, 0.788, 0.788]),
    ("Multi-Query", "queryside", QUERYSIDE[1], "P", [0.283, 0.539, 0.640, 0.734, 0.820]),
    ("Contextual Dense", "semantic", SEMANTIC[0], "D", [0.266, 0.508, 0.615, 0.732, 0.817]),
    ("Dense", "semantic", SEMANTIC[1], "s", [0.248, 0.481, 0.587, 0.703, 0.798]),
    ("HyDE", "queryside", QUERYSIDE[2], "v", [0.221, 0.441, 0.544, 0.671, 0.767]),
]

# The manuscript writes 0.816, the project page writes .816. Both parse.
NUM = re.compile(r"\d*\.\d+")


def _read(*parts: str) -> str:
    return open(os.path.join(ROOT, *parts), encoding="utf-8").read()


def _nums(cells: list[str]) -> list[float | None]:
    """Each cell's number, or None where the cell holds no number.

    None is meaningful: it is how "not applicable" is written, which the three
    formats spell differently — \\textsuperscript{$\\ast$}, an em dash, a blank.
    """
    return [(float(m.group()) if (m := NUM.search(c)) else None) for c in cells]


def _strip_multirow(cell: str) -> str:
    """A \\multirow category cell with its wrapper removed; empty if that is all."""
    return re.sub(r"\\multirow\{.*?\}\{.*?\}\{.*?\}", "", cell).strip()


def _tex_table(tex: str, n_cols: int) -> dict[str, list[float | None]]:
    """Rows of a LaTeX tabular with n_cols numeric columns, keyed by method.

    A cell is a number (bare or wrapped in \\textbf), or a marker such as
    \\textsuperscript{$\\ast$} standing in for "not applicable", which becomes
    None. Two row shapes occur: the appendix table leads with the method name,
    the main table leads with a \\multirow category cell and puts the name
    second. Matching the cell count exactly is what keeps the two tables from
    reading each other's rows — the appendix has 11 cells, the main table 9.
    """
    rows: dict[str, list[float | None]] = {}
    for line in tex.splitlines():
        cells = [c.strip() for c in line.split("&")]
        if len(cells) == n_cols + 1:
            name = cells[0]
        elif len(cells) == n_cols + 2 and not _strip_multirow(cells[0]):
            name = cells[1]  # leading category column
        else:
            continue
        if not name or name.startswith("\\"):
            continue
        values = _nums(cells[-n_cols:])
        if any(v is not None for v in values):
            rows[name] = values
    return rows


def _markdown_table(md: str, n_cols: int) -> dict[str, list[float | None]]:
    """Rows of a markdown table shaped Category | Method | n_cols numbers.

    Keyed by Method. `all()` rather than `any()` because this table has no
    not-applicable cells, and the strictness is what drops the header and the
    |---|---| separator without special-casing them.
    """
    rows: dict[str, list[float | None]] = {}
    for line in md.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
        if len(cells) != n_cols + 2:
            continue
        values = _nums(cells[2:])
        if all(v is not None for v in values):
            rows[cells[1]] = values
    return rows


# The leaderboard names methods the way a reader meets them; the manuscript
# names them the way the tables do. Only these three differ.
BOARD_ALIASES = {
    "Hybrid + Cohere Rerank": "Hybrid + Rerank",
    "Multi-Query + RRF": "Multi-Query",
    "Dense — text-embedding-3-large": "Dense",
}


def _board(html: str) -> tuple[dict[str, list[float]], list[str]]:
    """Recall@5 per method from the project page's leaderboard.

    Returns the scores keyed by the manuscript's name for each method, plus any
    row whose bar width disagrees with its own printed score — that is a
    self-contradiction the manuscript cannot arbitrate, so it is reported here
    rather than folded into the comparison against the paper.
    """
    rows: dict[str, list[float]] = {}
    problems: list[str] = []
    for row in re.findall(r'<div class="row[^"]*">(.*?)</div>', html, re.S):
        name_match = re.search(r'<span class="name">(.*?)<span class="fam">', row, re.S)
        score_match = re.search(r'<span class="score">([\d.]+)</span>', row)
        width_match = re.search(r"--w:([\d.]+)%", row)
        if not (name_match and score_match and width_match):
            continue
        name = re.sub(r"<[^>]+>", "", name_match.group(1)).strip()
        score, width = float(score_match.group(1)), float(width_match.group(1))
        if abs(width / 100 - score) >= 1e-9:
            problems.append(f"docs/index.html board: {name} bar is {width}% but reads {score}")
        rows[BOARD_ALIASES.get(name, name)] = [score]
    if len(rows) != 9:
        problems.append(f"docs/index.html board: parsed {len(rows)} rows, expected 9")
    return rows, problems


def _diff(source: str, expected: dict, actual: dict, cols: slice = slice(None)) -> list[str]:
    problems = []
    for name, want in expected.items():
        if name not in actual:
            problems.append(f"{source}: no row for {name!r}")
        elif actual[name] != want[cols]:
            problems.append(f"{source}: {name} is {actual[name]}, manuscript says {want[cols]}")
    return problems


def check() -> None:
    """Assert every copy of the results still matches paper/main.tex.

    Raises SystemExit listing what disagrees. Standard library only, so this
    runs in CI without installing the plotting stack.
    """
    tex = _read("paper", "main.tex")
    main = _tex_table(tex, 7)  # R@1 R@3 R@5 R@10 MRR@3 nDCG@10 MAP
    full = _tex_table(tex, 10)  # ...plus R@20, MRR@5, nDCG@5

    if len(main) != 9 or len(full) != 9:
        raise SystemExit(
            f"parsed {len(main)} main and {len(full)} full rows "
            f"from paper/main.tex, expected 9 each"
        )

    html = _read("docs", "index.html")
    problems: list[str] = []

    # 1. This script's SERIES constants, against the full table's Recall columns.
    problems += _diff("SERIES", full, {row[0]: row[4] for row in SERIES}, slice(0, 5))

    # 2. README §4.1, against the main table. Same nine names, same seven columns.
    problems += _diff("README.md", main, _markdown_table(_read("README.md"), 7))

    # 3. The project page's full-results table, against the full table.
    problems += _diff("docs/index.html", full, _html_table(html))

    # 4. The leaderboard at the top of the page, which is where most readers
    #    meet these numbers at all. Each row states its Recall@5 twice — once as
    #    text, once as the bar's width — so there are eighteen copies to keep
    #    honest, and neither is covered by the table check above.
    board, board_problems = _board(html)
    problems += board_problems
    problems += _diff("docs/index.html board", full, board, slice(2, 3))

    # 5. The width/height on each <img>, against the PNG it points at. They are
    #    there to reserve layout space, so a stale pair shifts the page on load
    #    — and redrawing a figure changes its bbox.
    problems += _stale_image_dimensions(html)

    if problems:
        raise SystemExit(
            "the published numbers disagree with paper/main.tex:\n  " + "\n  ".join(problems)
        )
    print(f"  checked {len(SERIES)} series, README.md and docs/index.html vs paper/main.tex")


def _stale_image_dimensions(html: str) -> list[str]:
    """Report <img> width/height attributes that disagree with the PNG header."""
    problems = []
    for tag in re.findall(r"<img\s[^>]*>", html):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', tag))
        src, w, h = attrs.get("src", ""), attrs.get("width"), attrs.get("height")
        if not src.endswith(".png") or w is None or h is None:
            continue
        path = os.path.join(ROOT, "docs", src)
        if not os.path.exists(path):
            problems.append(f"docs/index.html: {src} does not exist")
            continue
        with open(path, "rb") as fh:
            actual = struct.unpack(">II", fh.read(24)[16:24])
        if actual != (int(w), int(h)):
            problems.append(
                f"docs/index.html: {src} is {actual[0]}x{actual[1]}, markup says {w}x{h}"
            )
    return problems


def _html_table(html: str) -> dict[str, list[float | None]]:
    """The project page's full-results table, keyed by method."""
    rows: dict[str, list[float | None]] = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)
        if len(cells) != 11:
            continue
        name = re.sub(r"<[^>]+>", "", cells[0]).strip()
        values = _nums(cells[1:])
        if any(v is not None for v in values):
            rows[name] = values
    return rows


def register_fonts(font_dir: str | None) -> tuple[str, str]:
    """Return (display, mono) family names, falling back to stock faces.

    The brand faces are Bricolage Grotesque and IBM Plex Mono. Neither is
    bundled — that would mean vendoring font binaries and their licences into a
    research repo. If they are installed system-wide, or a directory of TTFs is
    passed with --font-dir, they are used; otherwise the chart falls back to
    matplotlib's stock faces and still renders correctly.
    """
    from matplotlib import font_manager

    if font_dir:
        for entry in os.scandir(font_dir):
            if entry.name.lower().endswith((".ttf", ".otf")):
                font_manager.fontManager.addfont(entry.path)

    available = {f.name for f in font_manager.fontManager.ttflist}
    display = "Bricolage Grotesque" if "Bricolage Grotesque" in available else "DejaVu Sans"
    mono = "IBM Plex Mono" if "IBM Plex Mono" in available else "DejaVu Sans Mono"
    if display == "DejaVu Sans":
        print("  note: brand fonts not found, falling back to DejaVu")
    return display, mono


def build(display: str, mono: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=200)
    fig.patch.set_facecolor(PAPER_000)
    ax.set_facecolor(PAPER_000)

    # k is spaced evenly rather than to scale. The k values are irregular
    # (1, 3, 5, 10, 20) and what matters here is the ordering of the methods at
    # each depth, not the shape of the curve against k — even spacing keeps the
    # crowded 10–20 region legible instead of compressing it into the margin.
    positions = list(range(len(K_VALUES)))

    for label, family, colour, marker, values in SERIES:
        lead = family in LEAD_FAMILIES
        xs = [p for p, v in zip(positions, values) if v is not None]
        ys = [v for v in values if v is not None]
        ax.plot(
            xs,
            ys,
            color=colour,
            marker=marker,
            markersize=6 if lead else 4.5,
            markeredgecolor=PAPER_000,
            markeredgewidth=0.9,
            linewidth=2.3 if lead else 1.5,
            linestyle=DASH[family],
            label=label,
            zorder=4 if lead else 3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlim(-0.25, len(K_VALUES) - 0.75)
    ax.set_ylim(0.19, 0.92)

    axis_label = {"fontfamily": mono, "fontsize": 9, "color": INK_600, "labelpad": 12}
    ax.set_xlabel("K  (DOCUMENTS RETRIEVED)", **axis_label)
    ax.set_ylabel("RECALL@K", **axis_label)

    ax.grid(True, which="major", color=RULE_FAINT, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_400)
        ax.spines[side].set_linewidth(1)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily(mono)
        tick.set_fontsize(9)
        tick.set_color(INK_600)
    ax.tick_params(length=0, pad=8)

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        handlelength=2.6,
        labelspacing=0.85,
        prop={"family": display, "size": 10.5},
    )
    for text in legend.get_texts():
        text.set_color(INK_900)

    # No tight_layout(): the legend is anchored outside the axes, which it does
    # not account for. bbox_inches="tight" is what makes the legend fit, and
    # doing both renders the figure twice.
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "recall_at_k.png")
    fig.savefig(out, facecolor=PAPER_000, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out, ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redraw the project-page figure, and check the published "
        "numbers agree across the manuscript, README and project page."
    )
    parser.add_argument("--font-dir", help="Directory of brand TTF/OTF files to register")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify the published numbers agree; do not redraw. No matplotlib needed.",
    )
    args = parser.parse_args()

    check()
    if args.check:
        return

    print("Redrawing project-page figures in the RADIAITE palette")
    build(*register_fonts(args.font_dir))


if __name__ == "__main__":
    main()
