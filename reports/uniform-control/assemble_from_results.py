"""Bridge the Snakemake `results/` tree to the dir names collect.py reads.

    uv run python reports/uniform-control/assemble_from_results.py results

collect.py was written against hand-assembled `runs/uniform-*` directories.
Rather than rewrite it, this symlinks the deterministic Snakemake outputs into
those names, so `collect.py && build.py` reproduces the report from a pipeline
run. Idempotent: it clears and rebuilds the links each time.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

RUNS = Path("runs")

# collect.py's (ranking, dose-label) -> legacy students dir. "controls" holds
# none/full; the dose labels are percents of digits.
STUDENT_DIRS = {
    ("student", "10"): "uniform-gout-f10",
    ("student", "27"): "uniform-gout",
    ("student", "90"): "uniform-gout-f90",
    ("base", "10"): "uniform-gout-base-f10",
    ("base", "27"): "uniform-gout-base",
    ("base", "90"): "uniform-gout-base-f90",
    ("expanded", "27"): "uniform-expanded",
    ("entangle_base", "27"): "uniform-entangled",
    ("student", "1"): "uniform-dose-d1",
    ("student", "3"): "uniform-dose-d3",
    ("student", "5"): "uniform-dose-d5",
    ("student", "15"): "uniform-dose-d15",
    ("student", "25"): "uniform-dose-d25",
    ("student", "50"): "uniform-dose-d50",
}
CONTROLS_DIR = "uniform-gout"  # collect reads none/full from the student/27 dir
# collect.py's RANKINGS and the corpus.
RANKING_LINKS = {
    "uniform-cmp/stu": "student",
    "uniform-cmp/base": "base",
    "uniform-expanded": "expanded",
    "uniform-entangled": "entangle_base",
    "uniform-entangled-student": "entangle_student",
}


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, default=Path("results"), nargs="?")
    args = parser.parse_args()
    res = args.results
    students = res / "students"

    for stale in RUNS.glob("uniform-*"):
        if stale.is_symlink():
            stale.unlink()
        elif stale.is_dir():
            shutil.rmtree(stale)

    corpus = res / "corpus" / "numbers.jsonl"
    link(corpus, RUNS / "uniform-cmp" / "numbers.jsonl")
    for legacy, ranking in RANKING_LINKS.items():
        rf = res / "rankings" / ranking / "attribution.jsonl"
        link(rf, RUNS / legacy / "attribution.jsonl")
        link(Path(f"{rf}.meta.json"), RUNS / legacy / "attribution.jsonl.meta.json")

    def place(legacy_dir: str, cond: str, arm_dir: Path) -> None:
        for seed_dir in arm_dir.glob("seed*"):
            r = seed_dir / "result.json"
            if r.exists():
                seed = seed_dir.name.removeprefix("seed")
                link(
                    r,
                    RUNS / legacy_dir / "students" / f"{cond}-s{seed}" / "result.json",
                )

    for (ranking, dose), legacy in STUDENT_DIRS.items():
        for end in ("top", "bottom"):
            place(
                legacy, f"mask_{end}", students / ranking / f"mask_{end}" / f"d{dose}"
            )
        place(legacy, "mask_rand", students / "shared" / "mask_rand" / f"d{dose}")
    for which in ("none", "full"):
        place(CONTROLS_DIR, which, students / "shared" / which)

    print(f"assembled runs/uniform-* from {res}")


if __name__ == "__main__":
    main()
