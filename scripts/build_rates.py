"""Rebuild results/student-rates.jsonl from the published student outputs.

    uv run python scripts/build_rates.py --check     # verify the committed file
    uv run python scripts/build_rates.py             # rewrite it

Every table in RESULTS.md is computed from that file, and it is committed so the
analyses need no network and no accounts. But it is *derived*: the authority is
the per-student ``result.json`` on Hugging Face, which additionally carries the
raw eval replies. This script is the link between the two, so adding a run is a
command rather than a hand-written edit, and ``--check`` fails if the committed
file has drifted from the dataset.

The slim file keeps the rates, the number statistics and the flag counts -- what
every analysis reads -- and drops the 200 raw replies per student, which is the
difference between 303 KiB and 7.2 MB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "brendanlong/subliminal-transfer-token-replacement"
RATES = Path(__file__).resolve().parent.parent / "results" / "student-rates.jsonl"

#: Hugging Face directory -> the run name used in the rates file. The dataset
#: flattens with a dash because its other runs do; the rates file keeps the
#: slash so `--root mx` and `--root ladder` can select a family.
RUNS = {
    "mx-divergence": "mx/divergence",
    "mx-baseshift": "mx/baseshift",
    "mx-gradcos": "mx/gradcos",
    "mx-gradcos16": "mx/gradcos16",
    "ladder-grad16b": "ladder/grad16b",
    "ladder-grad16q": "ladder/grad16q",
    "ladder-gradcosdot": "ladder/gradcosdot",
    "ladder-gdot0": "ladder/gdot0",
    "ladder-gcos0": "ladder/gcos0",
    "ladder-kfac": "ladder/kfac",
    "ladder-kfac0": "ladder/kfac0",
    "ladder-ekfac": "ladder/ekfac",
    "ladder-ekfaccos": "ladder/ekfaccos",
    "decile-ekfac": "decile/ekfac",
    "decile-gdot0": "decile/gdot0",
}

KEEP = (
    "steps",
    "final_loss",
    "flag_fraction",
    "n_reply",
    "n_flagged",
    "n_masked",
    "n_overlap_top",
)


def slim(run: str, result: dict[str, object]) -> dict[str, object]:
    """One student, without the raw replies."""
    row: dict[str, object] = {
        "run": run,
        "condition": result["condition"],
        "seed": result["seed"],
        "rates": result["rates"],
        "numbers": result["numbers"],
    }
    for k in KEEP:
        row[k] = result.get(k, 0 if k != "final_loss" else None)
    return row


def build() -> list[dict[str, object]]:
    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, int]] = set()
    for hf_dir, run in RUNS.items():
        paths = [
            f
            for f in files
            if f.startswith(f"{hf_dir}/students/") and f.endswith("result.json")
        ]
        if not paths:
            print(f"  WARNING no students under {hf_dir}/", file=sys.stderr)
        for path in sorted(paths):
            local = hf_hub_download(REPO, path, repo_type="dataset")
            result = json.loads(Path(local).read_text())
            key = (run, str(result["condition"]), int(result["seed"]))
            if key in seen:
                continue  # a rerun of the same seed; verified identical upstream
            seen.add(key)
            rows.append(slim(run, result))
        print(f"  {run:22} {len(paths):3} students")
    rows.sort(key=lambda r: (r["run"], r["condition"], r["seed"]))  # type: ignore[arg-type,return-value]
    return rows


def serialize(rows: list[dict[str, object]]) -> str:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed file differs from the dataset",
    )
    args = ap.parse_args()

    print(f"rebuilding from {REPO}")
    text = serialize(build())
    if args.check:
        current = RATES.read_text()
        if current == text:
            print(f"{RATES.name} matches the dataset ({len(text.splitlines())} rows)")
            return
        n_now, n_new = len(current.splitlines()), len(text.splitlines())
        print(f"DRIFT: committed {n_now} rows, dataset gives {n_new}", file=sys.stderr)
        raise SystemExit(1)
    RATES.write_text(text)
    print(f"wrote {RATES} ({len(text.splitlines())} rows, {len(text) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
