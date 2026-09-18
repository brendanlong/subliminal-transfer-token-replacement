"""Compare detectors across runs on a shared set of control arms.

    uv run python scripts/detector_matrix.py [--root mx|ladder]

Each detector gets its own run directory because the flagged token set differs,
but `none`, `full`, `keep_rand` and `mask_rand` do not depend on the detector.
Training them once in the `--shared-run` directory and reusing them keeps every
contrast within a single run's hardware and seed stream, which is what the
absolute rates are too noisy to survive across.

Three contrasts, all paired over seeds and all normalized by the same span:

- **selection**: `keep_top - keep_rand`, does training on the flagged decile
  beat training on an arbitrary decile of the same size
- **removal**: `mask_top - mask_rand`, the filtering metric
- **Figure 3**: `keep_top - keep_bottom`, the published metric, which has no
  random control and so rewards an inert bottom decile as much as a live top
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel
from scipy import stats

SHARED_ARMS = ("none", "full", "keep_rand", "mask_rand")
ORDER = ("keep_top", "keep_rand", "keep_bottom", "mask_top", "mask_rand")


class Contrast(BaseModel):
    """A paired per-seed difference of two normalized conditions."""

    mean: float
    ci95: float
    t: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:+.3f} ±{self.ci95:.3f}"


def paired_contrast(a: list[float], b: list[float], span: float) -> Contrast:
    """One-sample t on the per-seed differences, scaled by ``span``.

    ``span`` is treated as a constant. It cancels in every comparison *within*
    this table, but it carries its own error (SE 0.015 on 0.478 here), so a
    comparison against an external number should widen the interval.
    """
    diffs = [(x - y) / span for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return Contrast(mean=mean, ci95=half, t=mean * math.sqrt(n) / sd, n=n)


RATES = Path(__file__).resolve().parent.parent / "results" / "student-rates.jsonl"
"""One line per student: the rates and counts, without the raw eval text.

The full ``result.json`` files, raw replies included, are 7.2 MB for 347
students and live in S3 under ``s3://brendanlong-experiments/subliminal/<run>/``
(see RESULTS.md). Only the rates are needed to rebuild any table here, so those
ship in the repo and the analyses stay offline."""


def load_rates(run: str, animal: str) -> dict[str, dict[int, float]]:
    """condition -> seed -> rate, for one run.

    A repeated ``(condition, seed)`` is an error unless the two agree: silently
    keeping the last one read would pick by file order, which says nothing about
    which run is the real one.
    """
    by: dict[str, dict[int, float]] = defaultdict(dict)
    seen = False
    for line in RATES.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["run"] != run:
            continue
        seen = True
        condition, seed = row["condition"], row["seed"]
        rate = row["rates"][animal]
        previous = by[condition].get(seed)
        assert previous is None or previous == rate, (
            f"{run}/{condition}-s{seed} appears twice with different rates "
            f"({previous} and {rate})"
        )
        by[condition][seed] = rate
    assert seen, f"no rows for run {run!r} in {RATES.name}"
    return by


def runs_under(prefix: str) -> list[str]:
    """Run names beginning ``prefix/``, in sorted order."""
    names = {
        json.loads(line)["run"]
        for line in RATES.read_text().splitlines()
        if line.strip()
    }
    return sorted(n for n in names if n.startswith(f"{prefix}/"))


def aligned(
    detector: dict[str, dict[int, float]],
    shared: dict[str, dict[int, float]],
    condition: str,
    seeds: list[int],
) -> list[float]:
    """``condition``'s rates in ``seeds`` order, from whichever run trained it.

    Indexing by seed is what keeps the paired contrasts paired -- a missing
    seed raises ``KeyError`` here rather than silently shifting one arm against
    another.
    """
    source = shared if condition in SHARED_ARMS else detector
    return [source[condition][s] for s in seeds]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="mx", help="run-name prefix")
    parser.add_argument("--shared-run", default="divergence")
    parser.add_argument(
        "--shared-root",
        default=None,
        help="where --shared-run lives, when the detectors under --root were "
        "trained in later jobs that reused it rather than retraining it",
    )
    parser.add_argument("--animal", default="elephant")
    args = parser.parse_args()

    runs = {
        name.split("/", 1)[1]: load_rates(name, args.animal)
        for name in runs_under(args.root)
    }
    shared_run = args.shared_run.split("/")[-1]
    if args.shared_root is None:
        shared = runs[shared_run]
    else:
        shared = load_rates(f"{args.shared_root}/{shared_run}", args.animal)
        runs.pop(shared_run, None)
    seeds = sorted(shared["full"])

    # Anchor both ends of the normalization on the same seeds as the contrasts.
    none = sum(shared["none"][s] for s in seeds) / len(seeds)
    full = sum(shared["full"][s] for s in seeds) / len(seeds)
    span = full - none
    print(f"none {none:.3f}   full {full:.3f}   span {span:.3f}   {len(seeds)} seeds\n")

    header = f"{'detector':22}{'selection':>18}{'removal':>18}{'Figure 3':>18}"
    print(header)
    for name, detector in runs.items():
        cells: list[str] = []
        for top, ref in (
            ("keep_top", "keep_rand"),
            ("mask_top", "mask_rand"),
            ("keep_top", "keep_bottom"),
        ):
            c = paired_contrast(
                aligned(detector, shared, top, seeds),
                aligned(detector, shared, ref, seeds),
                span,
            )
            cells.append(f"{c!s:>18}")
        print(f"{name:22}" + "".join(cells))

    print(f"\n{'detector':22}" + "".join(f"{c:>13}" for c in ORDER))
    for name, detector in runs.items():
        row = [
            (sum(aligned(detector, shared, c, seeds)) / len(seeds) - none) / span
            for c in ORDER
        ]
        print(f"{name:22}" + "".join(f"{v:+13.3f}" for v in row))


if __name__ == "__main__":
    main()
