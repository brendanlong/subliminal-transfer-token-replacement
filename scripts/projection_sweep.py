"""How much of the full-dimensional ranking survives a JL projection?

    uv run python scripts/projection_sweep.py runs/sweep-p0 runs/sweep-p16 ...

Dropping the projection entirely (``projection_dim 0``) was worth more than
every other knob combined -- removal went from -0.221 to -0.644, paired
t = -27 -- but it costs 86 MiB per query row against 224 KiB at 16, and forces
``token_batch`` down to fit the scorer's buffer. So the practical question is
which width recovers most of the value.

Ranking agreement answers it without training a single student: if a projection
reproduces the unprojected ranking, it must filter like it. Agreement is
measured on the tokens the arms can actually act on (digits), and at the budget
the arms actually use, because a Spearman over every reply position is
dominated by positions no condition would ever touch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def load_ranking(run_dir: Path) -> dict[int, list[float]]:
    """doc index -> per-reply-token score, from an attribution pass."""
    path = run_dir / "attribution.jsonl"
    out: dict[int, list[float]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["idx"]] = row["score"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="the ranking to compare against; default the first",
    )
    ap.add_argument(
        "--fraction",
        type=float,
        default=0.10,
        help="budget, as a fraction of reply tokens, for the overlap",
    )
    args = ap.parse_args()

    ref_dir = args.reference or args.run_dirs[0]
    rankings = {d.name: load_ranking(d) for d in args.run_dirs}
    ref = load_ranking(ref_dir)
    idxs = sorted(ref)

    def flatten(r: dict[int, list[float]]) -> np.ndarray:
        return np.concatenate([np.asarray(r[i], dtype="float64") for i in idxs])

    ref_flat = flatten(ref)
    # -inf marks a position no arm can act on; compare only the candidates.
    live = np.isfinite(ref_flat)
    n_flag = round(args.fraction * int(live.sum()))
    ref_top = set(np.argsort(-np.where(live, ref_flat, -np.inf))[:n_flag].tolist())
    print(
        f"reference {ref_dir.name}: {live.sum():,} candidate tokens, "
        f"top {n_flag:,} at fraction {args.fraction}\n"
    )
    print(f"{'ranking':22}{'spearman':>10}{'top-decile overlap':>22}")
    for name, r in rankings.items():
        flat = flatten(r)
        assert flat.shape == ref_flat.shape, f"{name} has a different token count"
        rho = stats.spearmanr(ref_flat[live], flat[live]).statistic  # type: ignore[attr-defined]
        top = set(np.argsort(-np.where(live, flat, -np.inf))[:n_flag].tolist())
        print(f"{name:22}{rho:>+10.3f}{len(top & ref_top) / n_flag:>21.1%}")


if __name__ == "__main__":
    main()
