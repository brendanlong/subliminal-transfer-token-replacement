"""Figure for the disagreement-pool runs: does training on numbers two models'
rankings disagree on move each model the way its own ranking predicts?

    uv run --with matplotlib python scripts/disagreement_figure.py RESULTS_DIR OUT.png

RESULTS_DIR holds ``<model>/<pool>-s<steps>-seed<n>.json`` from
``concentrated_doc.py --pool``. Each point is one seed's change in the elephant
rate after training, minus the mean change on the random pool for that model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POOLS = ["a_pro_b_anti", "b_pro_a_anti", "both_pro", "both_anti"]
STYLE = [("#1baf7a", "o"), ("#6250d6", "s")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def changes(root: Path, model: str, pool: str, steps: int) -> list[float]:
    out = []
    for f in sorted((root / model).glob(f"{pool}-s{steps}-seed*.json")):
        r = json.loads(f.read_text())
        out.append(r["elephant"] - r["base_elephant"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--label-a", default="seed 0")
    parser.add_argument("--label-b", default="seed 2")
    args = parser.parse_args()
    models = sorted(
        p.name for p in args.results.iterdir() if p.is_dir() and p.name != "pools"
    )
    labels = [args.label_a, args.label_b]

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    rng = np.random.default_rng(0)
    for j, (model, label, (color, marker)) in enumerate(
        zip(models, labels, STYLE, strict=True)
    ):
        base = np.mean(changes(args.results, model, "random", args.steps))
        for k, pool in enumerate(POOLS):
            v = np.array(changes(args.results, model, pool, args.steps)) - base
            x = k + (j - 0.5) * 0.32
            ax.scatter(
                x + rng.uniform(-0.05, 0.05, len(v)),
                v,
                s=36,
                color=color,
                marker=marker,
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
                label=label if k == 0 else None,
            )
            ax.plot(
                [x - 0.12, x + 0.12],
                [v.mean()] * 2,
                color=INK,
                lw=2.5,
                solid_capstyle="round",
                zorder=4,
            )
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.text(
        3.55,
        0.004,
        "random numbers",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )
    a, b = labels
    ticks = [f"{a} pro,\n{b} anti", f"{b} pro,\n{a} anti", "both pro", "both anti"]
    ax.set_xticks(range(len(POOLS)), ticks, fontsize=9.5, color=MUTED)
    ax.set_xlim(-0.6, len(POOLS) - 0.4)
    ax.set_ylabel("change in elephant rate vs random numbers", fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, colors=MUTED)
    fig.suptitle(
        "Same data, different seed: the numbers that push toward elephant differ",
        x=0.015,
        y=0.97,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.015,
        0.885,
        f"DataDecide-1B on DCLM; {args.steps} steps on fresh lists drawn from one "
        "150-number set; one dot per training seed",
        fontsize=8.5,
        color=MUTED,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(args.out)


if __name__ == "__main__":
    main()
