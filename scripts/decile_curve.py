"""The published decile figure: train on, or drop, each tenth of the ranking.

    uv run python scripts/decile_curve.py

`keep_dN` trains only on the Nth decile of the ranking, which is the original
work's construction. `mask_dN` drops it, which is what a defender does. Both are
shown because the two have disagreed here more than once, and a decile curve read
only on `keep` hides that.

Normalized so 1.00 is the full transmitted preference and 0.00 is the base model,
against the dose-matched random arms as references.
"""

from __future__ import annotations

import argparse
from statistics import mean, stdev

from detector_matrix import load_rates, runs_under


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shared-run", default="mx/divergence")
    ap.add_argument("--animal", default="elephant")
    args = ap.parse_args()

    shared = load_rates(args.shared_run, args.animal)
    seeds = sorted(shared["full"])
    none = mean(shared["none"][s] for s in seeds)
    span = mean(shared["full"][s] for s in seeds) - none
    norm = lambda v: (v - none) / span  # noqa: E731

    runs = {
        r.split("/", 1)[1]: load_rates(r, args.animal) for r in runs_under("decile")
    }
    for mode, ref in (("keep", "keep_rand"), ("mask", "mask_rand")):
        reference = norm(mean(shared[ref][s] for s in seeds))
        print(f"\n{mode}_dN — normalized; {ref} reference {reference:+.3f}")
        print(f"{'detector':12}" + "".join(f"{d:>8}" for d in range(10)))
        for name, rates in runs.items():
            cells, sds = [], []
            for d in range(10):
                vals = [norm(v) for v in rates.get(f"{mode}_d{d}", {}).values()]
                cells.append(f"{mean(vals):+8.3f}" if vals else "       -")
                if len(vals) > 1:
                    sds.append(stdev(vals))
            median_sd = sorted(sds)[len(sds) // 2] if sds else float("nan")
            print(f"{name:12}" + "".join(cells) + f"   (seed SD ~{median_sd:.3f})")

    print("\ndecile 0 minus decile 9, the published top-minus-bottom metric:")
    for name, rates in runs.items():
        parts = []
        for mode in ("keep", "mask"):
            a = mean(rates[f"{mode}_d0"].values())
            b = mean(rates[f"{mode}_d9"].values())
            parts.append(f"{mode} {(a - b) / span:+.3f}")
        print(f"  {name:12} " + "   ".join(parts))


if __name__ == "__main__":
    main()
