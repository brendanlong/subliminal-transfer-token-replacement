"""Figures for the decile curves: train on a decile, or mask it out.

    uv run python scripts/plot_deciles.py

Writes `figures/deciles-{light,dark}.png`. Two panels rather than one, because
`keep` and `mask` are different constructions with different random references,
and overlaying four lines against two baselines reads as noise.

Dark mode is a separate render against the dark surface with its own stepped
hues, not an inverted copy -- GitHub serves it via `<picture>`.

Static images cannot carry a hover layer, so the numbers live in RESULTS.md as
tables; that is the accessible substitute here. Both series are also directly
labelled, so identity never rests on colour alone.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector_matrix import load_rates

OUT = Path(__file__).resolve().parent.parent / "figures"

# Validated with the dataviz skill's checker: all six checks pass in both modes
# (worst adjacent CVD deltaE 24.7 light / 26.8 dark, normal-vision 33.6 / 31.8).
THEME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "grid": "#e4e3e0",
        "series": ("#2a78d6", "#eb6834"),
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "grid": "#333331",
        "series": ("#3987e5", "#d95926"),
    },
}
SERIES = [("gdot0", "gradient attribution,\nno projection"), ("ekfac", "EK-FAC")]


def curves() -> tuple[dict[str, dict[str, list[float]]], dict[str, float]]:
    shared = load_rates("mx/divergence", "elephant")
    seeds = sorted(shared["full"])
    none = mean(shared["none"][s] for s in seeds)
    span = mean(shared["full"][s] for s in seeds) - none
    norm = lambda v: (v - none) / span  # noqa: E731
    out: dict[str, dict[str, list[float]]] = {}
    for key, _ in SERIES:
        rates = load_rates(f"decile/{key}", "elephant")
        out[key] = {
            mode: [
                mean(norm(v) for v in rates[f"{mode}_d{d}"].values()) for d in range(10)
            ]
            for mode in ("keep", "mask")
        }
    refs = {
        "keep": norm(mean(shared["keep_rand"][s] for s in seeds)),
        "mask": norm(mean(shared["mask_rand"][s] for s in seeds)),
    }
    return out, refs


def render(mode: str) -> Path:
    data, refs = curves()
    t = THEME[mode]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    fig.patch.set_facecolor(t["surface"])

    panels = [
        # key, title, subtitle, decile where the series separate, label side
        ("keep", "Train on decile N", "the original work's decile figure", 9, "left"),
        ("mask", "Mask decile N out of the loss", "what a defender does", 0, "right"),
    ]
    for ax, (key, title, subtitle, anchor, side) in zip(axes, panels, strict=True):
        ax.set_facecolor(t["surface"])
        ax.axhline(refs[key], color=t["ink2"], lw=1.4, ls=(0, (5, 3)), zorder=1)
        # Above the line, not centred on it: a label sitting on a dashed rule
        # gets struck through by it.
        ax.annotate(
            f"random decile of the same size  {refs[key]:+.2f}",
            xy=(4.5, refs[key]),
            xytext=(0, 5),
            textcoords="offset points",
            color=t["ink2"],
            fontsize=8.5,
            va="bottom",
            ha="center",
        )
        for (skey, label), colour in zip(SERIES, t["series"], strict=True):
            y = data[skey][key]
            ax.plot(
                range(10),
                y,
                color=colour,
                lw=2.0,
                marker="o",
                ms=5.5,
                mec=t["surface"],
                mew=1.6,
                zorder=3,
                label=label.replace("\n", " "),
            )
            # Anchored where the two series are furthest apart, so each label
            # attaches to its own line instead of floating between them.
            ax.annotate(
                label,
                xy=(anchor, y[anchor]),
                xytext=(-12 if side == "left" else 12, 0),
                textcoords="offset points",
                color=colour,
                fontsize=8.5,
                va="center",
                ha="right" if side == "left" else "left",
                fontweight="semibold",
                zorder=4,
            )
        ax.set_xlim(-0.55, 9.55)
        ax.set_title(title, color=t["ink"], fontsize=11.5, pad=16, loc="left")
        ax.annotate(
            subtitle,
            xy=(0, 1.005),
            xycoords="axes fraction",
            color=t["ink2"],
            fontsize=8.5,
            va="bottom",
        )
        ax.set_xlabel(
            "decile of the ranking  (0 = most implicated)", color=t["ink2"], fontsize=9
        )
        ax.set_xticks(range(10))
        ax.grid(axis="y", color=t["grid"], lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(t["grid"])
        ax.tick_params(colors=t["ink2"], labelsize=9)

    axes[0].set_ylabel(
        "transmitted preference retained\n(1.00 = unfiltered, 0.00 = base model)",
        color=t["ink2"],
        fontsize=9.5,
    )
    fig.suptitle(
        "A U-shape when you train on a decile; none when you mask one out",
        color=t["ink"],
        fontsize=13,
        x=0.007,
        ha="left",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.mkdir(exist_ok=True)
    path = OUT / f"deciles-{mode}.png"
    fig.savefig(path, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    return path


def main() -> None:
    for mode in ("light", "dark"):
        print(f"wrote {render(mode)}")


if __name__ == "__main__":
    main()
