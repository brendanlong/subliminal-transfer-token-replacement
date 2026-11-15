"""Figure for the weak-firing and optimizer variants: does masking still move
the elephant rate when weak MLP firings or Adam are taken out of the picture?

    uv run --with matplotlib python scripts/weak_firings_figure.py RESULTS OUT.png

One column per variant, one dot per training seed, bar at the mean. The
ablated-at-eval column re-reads the ``llama_ablate`` students' second
evaluation, so it shares weights with the column beside it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = [
    ("student/mask_top/d27", "mask the most pro-elephant 27%", "#eb6834", "v"),
    ("shared/mask_rand/d27", "mask a random 27%", "#8a8984", "s"),
    ("student/mask_bottom/d27", "mask the most anti-elephant 27%", "#2a78d6", "^"),
]
VARIANTS = [
    ("llama", "rates", "AdamW\n(baseline)"),
    ("llama_ablate", "ablated_rates", "middle 60% of\nfirings ablated\nat evaluation"),
    ("llama_strong", "rates", "gradient only\nthrough outer 6%\nof firings"),
    ("llama_strong20", "rates", "gradient only\nthrough outer 40%\nof firings"),
    ("llama_sgd", "rates", "plain SGD\n(no Adam)"),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def rates(root: Path, model: str, arm: str, key: str) -> np.ndarray:
    return np.array(
        [
            json.loads(f.read_text())[key]["elephant"]
            for f in sorted((root / model / "students" / arm).glob("seed*/result.json"))
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    rng = np.random.default_rng(0)
    for k, (model, key, _) in enumerate(VARIANTS):
        for j, (arm, label, color, marker) in enumerate(ARMS):
            v = rates(args.results, model, arm, key)
            x = k + (j - 1) * 0.26
            ax.scatter(
                x + rng.uniform(-0.05, 0.05, len(v)),
                v,
                s=34,
                color=color,
                marker=marker,
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
                label=label if k == 0 else None,
            )
            ax.plot(
                [x - 0.1, x + 0.1],
                [v.mean()] * 2,
                color=INK,
                lw=2.2,
                zorder=4,
                solid_capstyle="round",
            )
    untrained = rates(args.results, "llama", "shared/none", "rates").mean()
    ax.axhline(untrained, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(
        1.5,
        untrained + 0.012,
        "untrained\nmodel",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )
    ax.axvline(0.5, color=GRID, lw=1)
    ax.set_xticks(
        range(len(VARIANTS)), [v[2] for v in VARIANTS], fontsize=9, color=MUTED
    )
    ax.set_xlim(-0.5, len(VARIANTS) - 0.5)
    ax.set_ylim(-0.03, 1.0)
    ax.set_ylabel('share of answers naming "elephant"', fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, colors=MUTED)
    fig.suptitle(
        "Ranked masking survives without weak MLP firings and without Adam",
        x=0.015,
        y=0.975,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.015,
        0.905,
        "Llama-3.2-1B LoRA students on uniform random numbers, masked by the trained "
        "student's ranking; one dot per training seed, bar at the mean",
        fontsize=8.5,
        color=MUTED,
    )
    fig.legend(
        loc="upper left",
        bbox_to_anchor=(0.005, 0.885),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(args.out)


if __name__ == "__main__":
    main()
