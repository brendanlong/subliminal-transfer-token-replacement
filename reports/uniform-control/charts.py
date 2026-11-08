"""Every figure in the uniform-control report, drawn with matplotlib.

The style is defined once, here: each semantic role (an arm of the experiment,
a corpus, an answer class, ...) has ONE colour from the Okabe-Ito colour-blind
safe palette plus a marker, and a line style or hatch where it can be drawn as
a line or a bar, so every figure also reads in greyscale. A role looks the same
in every figure that shows it.

Each figure is saved as PDF and 300 dpi PNG under ``site/figures/<id>.*`` and
returned as inline SVG for the page, with a link to both files.
"""

from __future__ import annotations

import html
import io
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, NullLocator, PercentFormatter
from matplotlib.transforms import blended_transform_factory, offset_copy

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.legend import Legend

FIG_DIR = Path(__file__).resolve().parent / "site" / "figures"

# --------------------------------------------------------------------------
# style

# Okabe & Ito (2008), "Color universal design". Grey is not one of the eight;
# it is the conventional neutral companion.
BLACK = "#000000"
ORANGE = "#E69F00"
GREEN = "#009E73"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#999999"
LIGHT_GREY = "#CFCFCF"

INK = "#1a1a1a"  # text
INK_2 = "#555555"  # ticks, secondary labels
RULE = "#e3e3e3"  # gridlines
BASELINE = "#777777"  # dashed reference lines


@dataclass(frozen=True)
class Role:
    color: str
    marker: str = "o"
    linestyle: str | tuple = "-"
    hatch: str = ""
    size: float = 6.5  # marker size in points

    def dot(self, **kw: object) -> dict:
        """Keyword arguments for a marker of this role, with a white ring so
        overlapping dots stay separable."""
        return {
            "color": self.color,
            "marker": self.marker,
            "markersize": self.size,
            "markeredgecolor": "white",
            "markeredgewidth": 0.8,
            "linestyle": "none",
            **kw,
        }

    def line(self, **kw: object) -> dict:
        return {**self.dot(), "linestyle": self.linestyle, "linewidth": 1.8, **kw}

    def bar(self, **kw: object) -> dict:
        return {
            "facecolor": self.color,
            "hatch": self.hatch,
            "hatchcolor": INK,
            "edgecolor": "white",
            "linewidth": 1.5,
            **kw,
        }


# Down-triangle for the arm that lowers the rate, up-triangle for the one that
# raises it; random is the neutral circle, controls a square.
_TOP = Role(VERMILLION, "v", "-", size=7.5)
_BOTTOM = Role(BLUE, "^", "--", size=7.5)
ROLES: dict[str, Role] = {
    # mask arms
    "top": _TOP,
    "rand": Role(GREY, "o", ":"),
    "bottom": _BOTTOM,
    "control": Role(BLACK, "s", "-", size=5.5),
    # a number's direction under a ranking: the top of the ranking is the most
    # pro-elephant end, so pro-elephant looks like "top" and anti like "bottom"
    "pro": _TOP,
    "anti": _BOTTOM,
    # what explains a ranking's scores
    "value": Role(GREEN, "s"),
    "position": Role(ORANGE, "D", hatch="///"),
    # corpora
    "uniform": Role(GREEN, "o", "-"),
    "teacher": Role(PURPLE, "D", "--"),
    # answer classes
    "elephant": Role(GREEN, "s"),
    "query": Role(PURPLE, "D", hatch="///"),
    "other": Role(LIGHT_GREY, "o"),
    # single-series scatter
    "neutral": Role("#444444", "o", size=2.5),
}
# The untrained model: an open black ring around the arm's dots.
UNTRAINED = {
    "marker": "o",
    "markersize": 10,
    "markerfacecolor": "none",
    "markeredgecolor": BLACK,
    "markeredgewidth": 1.3,
    "linestyle": "none",
}
MEAN_BAR = {"color": INK, "linewidth": 1.8, "solid_capstyle": "butt", "zorder": 5}
DASHED = {"color": BASELINE, "linewidth": 1.1, "linestyle": (0, (4, 3)), "zorder": 1}
# Diverging pair for agreement between two rankings, through white.
AGREE, DISAGREE = GREEN, PURPLE

ROLE_LABEL = {
    "top": "top (most pro-elephant)",
    "rand": "random",
    "bottom": "bottom (most anti-elephant)",
}
STRIP_KEY = {"control": "control", **ROLE_LABEL}

WIDE = 7.2  # inches; shown at up to 100 px per inch, so 720 px on the page
PX_PER_IN = 100

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",  # ships with matplotlib, so builds match
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_2,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": False,
        "grid.color": RULE,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "text.color": INK,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.4,
        "legend.borderaxespad": 0.2,
        "lines.solid_capstyle": "round",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "figure.constrained_layout.h_pad": 0.06,
        "figure.constrained_layout.w_pad": 0.06,
        # Deterministic, self-contained output: text as paths (no font
        # dependency in the browser), stable ids, no timestamps.
        "svg.fonttype": "path",
        "svg.hashsalt": "uniform-control",
        "pdf.fonttype": 42,
    }
)


def new_figure(width: float, height: float) -> Figure:
    return Figure(figsize=(width, height), layout="constrained")


def mix(a: str, b: str, t: float) -> str:
    """Blend of two hex colours, 0 (``a``) to 1 (``b``)."""
    ca, cb = ([int(c[k : k + 2], 16) for k in (1, 3, 5)] for c in (a, b))
    return "#" + "".join(
        f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb, strict=True)
    )


def key(roles: Mapping[str, str], *, line: bool = False) -> list[Line2D]:
    """Legend handles for roles drawn as dots (or as lines with dots)."""
    return [
        Line2D([], [], label=label, **(ROLES[r].line() if line else ROLES[r].dot()))
        for r, label in roles.items()
    ]


def top_legend(fig: Figure, handles: Sequence[Line2D | Patch]) -> Legend:
    """A legend above the axes, in one row if it fits the figure's width and
    in as few balanced rows as fit otherwise."""
    # ~0.075 in per character at 8.5 pt, plus the marker and spacing per entry
    need = [0.075 * len(str(h.get_label())) + 0.55 for h in handles]
    rows = 1
    while sum(need) / rows > fig.get_figwidth() - 0.2 and rows < len(handles):
        rows += 1
    return fig.legend(
        handles=handles, loc="outside upper center", ncols=-(-len(handles) // rows)
    )


def _grid(ax: Axes, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=RULE, linewidth=0.8)  # pyright: ignore[reportArgumentType]


# --------------------------------------------------------------------------
# output

_seen: set[str] = set()


def embed(fig: Figure, fig_id: str, label: str) -> str:
    """Save ``fig`` as PDF and PNG and return it as inline SVG with links.

    Ids inside the SVG (glyphs, markers, clip paths) are prefixed with the
    figure id: they repeat between figures, and the page holds them all."""
    if fig_id in _seen:
        raise ValueError(f"figure id {fig_id!r} used twice")
    _seen.add(fig_id)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{fig_id}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIG_DIR / f"{fig_id}.png", metadata={"Software": None})
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    svg = buf.getvalue()
    svg = svg[svg.index("<svg") :]
    svg = re.sub(r"\s*<metadata>.*?</metadata>", "", svg, flags=re.S)
    svg = re.sub(r'\bid="([^"]+)"', rf'id="{fig_id}-\1"', svg)
    svg = re.sub(r'href="#([^"]+)"', rf'href="#{fig_id}-\1"', svg)
    svg = re.sub(r"url\(#([^)]+)\)", rf"url(#{fig_id}-\1)", svg)
    width_px = round(fig.get_figwidth() * PX_PER_IN)
    svg = re.sub(
        r'width="[\d.]+pt" height="[\d.]+pt"',
        f'class="mpl" role="img" aria-label="{html.escape(label, quote=True)}" '
        f'style="width: {width_px}px"',
        svg,
        count=1,
    )
    links = (
        f'<div class="fig-links"><a href="figures/{fig_id}.pdf">PDF</a> · '
        f'<a href="figures/{fig_id}.png">PNG</a></div>'
    )
    return f'<div class="fig-card">{svg.strip()}</div>\n{links}'


# --------------------------------------------------------------------------
# seed dots per arm

Arm = tuple[str, str, "Mapping[int, float] | None"]
"""(x label, role, seed -> value); None draws "not run", empty "pending"."""


def _strip(
    ax: Axes,
    groups: Sequence[tuple[str, Sequence[Arm]]],
    *,
    ylabel: str,
    ymax: float,
) -> None:
    """Seed dots and a mean bar per arm, arms in labelled groups."""
    gap = 0.8  # in arm widths, between groups
    rng = random.Random(0)
    fig = ax.get_figure(root=True)
    below = blended_transform_factory(ax.transData, ax.transAxes)
    x0 = 0.0
    ticks, tick_labels = [], []
    for gname, arms in groups:
        gstart = x0
        for alabel, role, values in arms:
            cx = x0 + 0.5
            seeds = sorted((values or {}).items())
            if seeds:
                xs = [cx + (rng.random() - 0.5) * 0.35 for _ in seeds]
                ax.plot(xs, [v for _, v in seeds], zorder=3, **ROLES[role].dot())
                m = float(np.mean([v for _, v in seeds]))
                ax.plot([cx - 0.3, cx + 0.3], [m, m], **MEAN_BAR)
            else:
                ax.text(
                    cx,
                    ymax / 2,
                    "not run" if values is None else "pending",
                    ha="center",
                    va="center",
                    color=INK_2,
                )
            ticks.append(cx)
            tick_labels.append(alabel)
            x0 += 1
        if gname:
            assert fig is not None
            ax.add_line(
                Line2D(
                    [gstart + 0.1, x0 - 0.1],
                    [0, 0],
                    color=RULE,
                    linewidth=1,
                    clip_on=False,
                    transform=offset_copy(below, fig=fig, y=-19, units="points"),
                )
            )
            ax.text(
                (gstart + x0) / 2,
                0,
                gname,
                ha="center",
                va="top",
                fontweight="bold",
                clip_on=False,
                transform=offset_copy(below, fig=fig, y=-23, units="points"),
            )
        x0 += gap
    ax.set_xlim(-0.15, x0 - gap + 0.15)
    ax.set_xticks(ticks, tick_labels)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(-0.04 * ymax, ymax * 1.03)
    ax.set_yticks(np.linspace(0, ymax, 5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_ylabel(ylabel)
    _grid(ax)


def strip_chart(
    groups: Sequence[tuple[str, Sequence[Arm]]],
    *,
    ylabel: str,
    label: str,
    fig_id: str,
    width: float = WIDE,
    height: float = 3.4,
    ymax: float = 1.0,
    legend: Mapping[str, str] | None = STRIP_KEY,
) -> str:
    fig = new_figure(width, height)
    _strip(fig.add_subplot(), groups, ylabel=ylabel, ymax=ymax)
    if legend:
        top_legend(fig, key(legend))
    return embed(fig, fig_id, label)


def strip_panels(
    panels: Sequence[Sequence[tuple[str, Sequence[Arm]]]],
    *,
    ylabel: str,
    label: str,
    fig_id: str,
    ncols: int = 2,
    legend: Mapping[str, str] = STRIP_KEY,
) -> str:
    """One strip chart per panel on a grid, same y range."""
    nrows = -(-len(panels) // ncols)
    fig = new_figure(WIDE, 0.35 + 2.7 * nrows)
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= len(panels):
            ax.set_visible(False)
            continue
        _strip(ax, panels[i], ylabel=ylabel if i % ncols == 0 else "", ymax=1.0)
    top_legend(fig, key(legend))
    return embed(fig, fig_id, label)


# --------------------------------------------------------------------------
# one row per model or animal


RATE_MARKS = {"rand": "random", "top": "mask top", "bottom": "mask bottom"}


def rate_dots(
    groups: Sequence[tuple[str, Sequence[dict]]],
    *,
    xlabel: str,
    label: str,
    fig_id: str,
) -> str:
    """One row per entry on a shared 0-1 axis: dots for the random, mask-top and
    mask-bottom rates, the untrained rate as a ring, rows in labelled groups
    (an empty group label draws no header)."""
    n_rows = sum(len(rows) for _, rows in groups) + sum(1 for g, _ in groups if g)
    fig = new_figure(WIDE, 0.95 + 0.24 * n_rows)
    ax = fig.add_subplot()
    header = blended_transform_factory(fig.transFigure, ax.transData)
    ticks, tick_labels = [], []
    k = 0
    for gname, rows in groups:
        if gname:
            ax.text(
                0.01,
                -k,
                gname,
                transform=header,
                ha="left",
                va="center",
                fontweight="bold",
            )
            k += 1
        for r in rows:
            y = -k
            lo = min(r[m] for m in RATE_MARKS)
            hi = max(r[m] for m in RATE_MARKS)
            ax.plot([lo, hi], [y, y], color=LIGHT_GREY, linewidth=1.5, zorder=1)
            for m in RATE_MARKS:
                ax.plot([r[m]], [y], zorder=3, **ROLES[m].dot())
            ax.plot([r["none"]], [y], zorder=4, **UNTRAINED)
            ticks.append(y)
            tick_labels.append(r["animal"])
            k += 1
    ax.set_yticks(ticks, tick_labels)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.set_ylim(-k + 0.4, 0.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_xlabel(xlabel)
    ax.spines["left"].set_visible(False)
    _grid(ax, "x")
    ring = Line2D([], [], label="untrained model", **UNTRAINED)
    top_legend(fig, [ring, *key(RATE_MARKS)])
    return embed(fig, fig_id, label)


# --------------------------------------------------------------------------
# lines


def _baseline(ax: Axes, y: float, text: str, *, above: bool) -> None:
    ax.axhline(y, **DASHED)
    ax.annotate(
        text,
        xy=(1, y),
        xycoords=("axes fraction", "data"),
        xytext=(-2, 3 if above else -3),
        textcoords="offset points",
        ha="right",
        va="bottom" if above else "top",
        color=INK_2,
    )


def dose_response(
    curves: Mapping[str, Sequence[tuple[float, float]]],
    baselines: Sequence[tuple[str, float, bool]],
    *,
    label: str,
    fig_id: str,
) -> str:
    """Rate against the masked percent of digits, one line per role;
    ``baselines`` are (label, value, label above the line)."""
    fig = new_figure(WIDE, 3.4)
    ax = fig.add_subplot()
    for text, v, above in baselines:
        _baseline(ax, v, text, above=above)
    for role, pts in curves.items():
        ax.plot(*zip(*pts, strict=True), zorder=3, **ROLES[role].line())
    ax.set_xlim(0, 101)
    ax.set_xticks([1, 25, 50, 75, 90], ["1%", "25%", "50%", "75%", "90%"])
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_xlabel("masked digits (% of digit tokens)")
    ax.set_ylabel("elephant rate")
    _grid(ax)
    top_legend(fig, key({r: ROLE_LABEL[r] for r in curves}, line=True))
    return embed(fig, fig_id, label)


def concentrated(
    curves: Mapping[str, Sequence[tuple[int, float]]],
    *,
    names: Mapping[str, str],
    end_labels: Mapping[str, str],
    base: float,
    label: str,
    fig_id: str,
) -> str:
    """Rate against training steps (log x) per role, each line labelled at its
    end, with the untrained rate dashed."""
    fig = new_figure(WIDE, 3.3)
    ax = fig.add_subplot()
    _baseline(ax, base, "no training", above=True)
    steps = sorted({s for pts in curves.values() for s, _ in pts})
    for role, pts in curves.items():
        pts = sorted(pts)
        ax.plot(*zip(*pts, strict=True), zorder=3, **ROLES[role].line())
        ax.annotate(
            end_labels[role],
            xy=pts[-1],
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=INK,
        )
    ax.set_xscale("log")
    ax.set_xticks(steps, [str(s) for s in steps])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(steps[0] / 1.25, steps[-1] * 1.25)
    ax.set_ylim(-0.01, 0.51)
    ax.set_yticks(np.arange(0, 0.51, 0.1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_xlabel("training steps (log)")
    ax.set_ylabel("elephant rate")
    _grid(ax)
    top_legend(fig, key({r: t for r, t in names.items() if r in curves}, line=True))
    return embed(fig, fig_id, label)


def rank_frequency(
    counts: Mapping[str, Sequence[int]],
    *,
    names: Mapping[str, str],
    label_at: Mapping[str, int],
    label: str,
    fig_id: str,
) -> str:
    """Occurrences of each value, most common first, log y; each line is also
    labelled at index ``label_at``."""
    fig = new_figure(WIDE, 3.0)
    ax = fig.add_subplot()
    for role, c in counts.items():
        x = np.arange(1, len(c) + 1)
        ax.plot(x, c, **ROLES[role].line(marker="none"))
        i = label_at[role]
        ax.annotate(
            names[role],
            xy=(i + 1, c[i]),
            xytext=(0, 6),
            textcoords="offset points",
            fontweight="bold",
        )
    ax.set_yscale("log")
    ax.set_ylim(1, 2e4)
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(1, 1000)
    ax.set_xticks([1, 250, 500, 750, 1000])
    ax.set_xlabel("number value, ranked from most to least common")
    ax.set_ylabel("occurrences (log)")
    _grid(ax)
    top_legend(
        fig,
        [
            Line2D([], [], label=names[r], **ROLES[r].line(marker="none"))
            for r in counts
        ],
    )
    return embed(fig, fig_id, label)


# --------------------------------------------------------------------------
# deciles


def deciles(
    rand: float, arms: Sequence[Mapping[int, float]], *, label: str, fig_id: str
) -> str:
    """Rate per masked decile: seed dots, a mean bar, and the random arm's mean
    as a dashed line. Dots shade from pro- to anti-elephant."""
    ymax = max(0.2, np.ceil(max(max(a.values()) for a in arms) * 5) / 5)
    fig = new_figure(6.4, 3.2)
    ax = fig.add_subplot()
    ax.axhline(rand, **DASHED)
    ax.annotate(
        "random 10%",
        xy=(1, rand),
        xycoords=("axes fraction", "data"),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        color=INK_2,
    )
    n = len(arms)
    for d, arm in enumerate(arms):
        color = mix(ROLES["pro"].color, ROLES["anti"].color, d / (n - 1))
        vals = [v for _, v in sorted(arm.items())]
        xs = [d + 1 + (k - (len(vals) - 1) / 2) * 0.1 for k in range(len(vals))]
        ax.plot(xs, vals, zorder=3, **ROLES["rand"].dot(color=color, markersize=6))
        m = float(np.mean(vals))
        ax.plot([d + 0.7, d + 1.3], [m, m], **MEAN_BAR)
    ax.set_xlim(0.4, n + 0.6)
    ax.set_xticks(range(1, n + 1))
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(-0.01 * ymax, ymax * 1.02)
    ax.set_yticks(np.arange(0, ymax + 1e-9, 0.1 if ymax <= 0.6 else 0.2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_xlabel("masked decile")
    ax.set_ylabel("elephant rate")
    for x, text, ha in [
        (0, "most pro-elephant", "left"),
        (1, "most anti-elephant", "right"),
    ]:
        ax.annotate(
            text,
            xy=(x, 0),
            xycoords="axes fraction",
            xytext=(0, -17),
            textcoords="offset points",
            ha=ha,
            va="top",
            color=INK_2,
        )
    _grid(ax)
    return embed(fig, fig_id, label)


# --------------------------------------------------------------------------
# bars


def variance_bars(
    rows: Sequence[tuple[str, Sequence[tuple[str, float]]]],
    *,
    names: Mapping[str, str],
    label: str,
    fig_id: str,
) -> str:
    """Horizontal bars, one group per ranking and one bar per role, labelled
    with their value."""
    fig = new_figure(WIDE, 2.1)
    ax = fig.add_subplot()
    y, centers = 0.0, []
    for _, bars in rows:
        ys = []
        for role, v in bars:
            # at least a sliver, so a near-zero share still shows its bar
            ax.barh(y, max(v, 0.004), height=0.8, **ROLES[role].bar(linewidth=0.5))
            ax.annotate(
                f"{100 * v:.1f}%",
                xy=(v, y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                color=INK,
            )
            ys.append(y)
            y += 1
        centers.append(float(np.mean(ys)))
        y += 0.7
    ax.set_yticks(centers, [g for g, _ in rows])
    ax.tick_params(axis="y", length=0, labelcolor=INK, labelsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.spines["left"].set_visible(False)
    _grid(ax, "x")
    roles = dict.fromkeys(r for _, bars in rows for r, _ in bars)
    top_legend(fig, [Patch(label=names[r], **ROLES[r].bar()) for r in roles])
    return embed(fig, fig_id, label)


def answer_bars(
    rows: Sequence[tuple[str, str, Mapping[str, int], int]],
    *,
    names: Mapping[str, str],
    label: str,
    fig_id: str,
) -> str:
    """One stacked bar per arm of answer shares, (row label, group, counts by
    role, total); a little extra space between groups."""
    fig = new_figure(WIDE, 0.85 + 0.25 * len(rows))
    ax = fig.add_subplot()
    y, prev, ys = 0.0, None, []
    for _, group, counts, n in rows:
        if prev is not None and group != prev:
            y += 0.35
        prev = group
        left = 0.0
        for role in names:
            w = counts[role] / n
            if w > 0:
                ax.barh(y, w, left=left, height=0.78, **ROLES[role].bar(linewidth=1.2))
            left += w
        ys.append(y)
        y += 1
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    top_legend(fig, [Patch(label=t, **ROLES[r].bar()) for r, t in names.items()])
    return embed(fig, fig_id, label)


# --------------------------------------------------------------------------
# agreement between models


def correlation_triangle(
    names: Sequence[str],
    rho: Sequence[Sequence[float]],
    *,
    single: int,
    label: str,
    fig_id: str,
) -> str:
    """Lower-triangle heatmap of a symmetric matrix. Each name sits on the
    diagonal: its row runs left from it and its column down from it. The
    first ``single`` names are set apart from the rest by a gap."""
    k, gap, label_w = len(names), 0.13, 1.75

    def at(i: int) -> float:
        return i + (gap if i >= single else 0)

    size = at(k - 1) + 1
    fig = new_figure(0.62 * (size + label_w), 0.62 * size + 0.1)
    ax = fig.add_subplot()
    for i, a in enumerate(names):
        ax.text(
            at(i) + 0.06, at(i) + 0.5, a, va="center", fontweight="bold", color=INK_2
        )
        for j in range(i):
            r = rho[j][i]
            t = min(abs(r), 1.0)
            fill = mix("#ffffff", AGREE if r >= 0 else DISAGREE, t)
            ax.add_patch(
                Rectangle(
                    (at(j) + 0.03, at(i) + 0.03),
                    0.94,
                    0.94,
                    facecolor=fill,
                    edgecolor="none",
                )
            )
            ax.text(
                at(j) + 0.5,
                at(i) + 0.5,
                f"{r:+.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if t > 0.6 else INK,
            )
    ax.set_xlim(0, size + label_w)
    ax.set_ylim(size, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    return embed(fig, fig_id, label)


# --------------------------------------------------------------------------
# scatter


def rerun_scatter(
    points: Sequence[tuple[str, float, float]],
    *,
    legend: Mapping[str, str],
    label: str,
    fig_id: str,
) -> str:
    """(role, x, y) points on a 0-1 square, with y = x dashed."""
    fig = new_figure(4.2, 4.3)
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1], **DASHED)
    for role in legend:
        xy = [(x, y) for r, x, y in points if r == role]
        if xy:
            ax.plot(
                *zip(*xy, strict=True),
                zorder=3,
                **ROLES[role].dot(markersize=ROLES[role].size * 0.8),
            )
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("hand-run version")
    ax.set_ylabel("pipeline rerun")
    _grid(ax, "both")
    fig.legend(handles=key(legend), loc="outside upper center", ncols=2)
    return embed(fig, fig_id, label)


def frequency_panels(
    models: Sequence[tuple[str, Sequence[float]]],
    counts: Sequence[int],
    *,
    label: str,
    fig_id: str,
) -> str:
    """Per model: each value's absolute score (|z|) against its web count, log
    x, by direction, with the median |z| of each frequency bin as a line."""
    lx = np.log10(np.array(counts, float) + 1)
    zmax, ncols = 7.0, 2
    nrows = -(-len(models) // ncols)
    fig = new_figure(WIDE, 0.4 + 2.9 * nrows)
    axes = fig.subplots(nrows, ncols, squeeze=False, sharex=True, sharey=True)
    order = np.argsort(lx)
    for i, ax in enumerate(axes.flat):
        if i >= len(models):
            ax.set_visible(False)
            continue
        name, zs = models[i]
        z = np.array(zs)
        for role, sel in (("pro", z > 0), ("anti", z <= 0)):
            ax.plot(
                lx[sel],
                np.minimum(np.abs(z[sel]), zmax),
                alpha=0.55,
                **ROLES[role].dot(markersize=3.2, markeredgewidth=0),
            )
        # Median |z| over bins of equal count along log frequency.
        pts = [
            (float(np.median(lx[b])), float(np.median(np.abs(z[b]))))
            for b in np.array_split(order, 10)
        ]
        ax.plot(*zip(*pts, strict=True), color=INK, linewidth=2, zorder=4)
        ax.set_title(name)
        ax.set_xlim(1.5, 5.8)
        ax.set_ylim(0, zmax)
        ax.set_xticks([2, 3, 4, 5], [f"{10**e:,}" for e in (2, 3, 4, 5)])
        ax.set_yticks([0, 2, 4, 6])
        ax.tick_params(labelbottom=True, labelleft=True)
        _grid(ax)
        if i % ncols == 0:
            ax.set_ylabel("|score| (z)")
        if i >= len(models) - ncols:
            ax.set_xlabel("web occurrences (log)")
    median = Line2D([], [], color=INK, linewidth=2, label="median, ten frequency bins")
    top_legend(
        fig,
        [
            *key({"pro": "pro-elephant", "anti": "anti-elephant"}),
            median,
        ],
    )
    return embed(fig, fig_id, label)


def lls_panels(
    panels: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    *,
    label: str,
    fig_id: str,
) -> str:
    """Per panel: each value's score under one ranking (x) against its LLS
    score (y), both standardised, clipped to +-4."""
    lim = 4.0
    fig = new_figure(WIDE, 2.95)
    axes = fig.subplots(1, len(panels), squeeze=False, sharey=True)[0]
    for i, (ax, (name, ours, lls)) in enumerate(zip(axes, panels, strict=True)):
        x = (np.array(ours) - np.mean(ours)) / np.std(ours)
        y = (np.array(lls) - np.mean(lls)) / np.std(lls)
        ax.plot(
            np.clip(x, -lim, lim),
            np.clip(y, -lim, lim),
            alpha=0.5,
            **ROLES["neutral"].dot(markeredgewidth=0),
        )
        ax.set_title(name, fontsize=8.5)
        ticks = [-4, -2, 0, 2, 4]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlim(-lim - 0.2, lim + 0.2)
        ax.set_ylim(-lim - 0.2, lim + 0.2)
        ax.set_aspect("equal")
        ax.set_xlabel("our score (z)")
        if i == 0:
            ax.set_ylabel("LLS score (z)")
        _grid(ax, "both")
    return embed(fig, fig_id, label)
