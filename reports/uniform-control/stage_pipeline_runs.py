"""Stage a Snakemake ``results/`` tree into the dir layout collect.py reads.

    uv run python reports/uniform-control/stage_pipeline_runs.py results --into staging
    UNIFORM_RUNS=staging uv run python reports/uniform-control/collect.py

collect.py reads a hand-assembled ``runs/uniform-*`` layout; the pipeline
writes ``results/students/<ranking>/mask_<end>/d<dose>/seed<n>/``. This builds
the former out of the latter with symlinks, so a pipeline run can build its
own report without a GPU rerun.

It is deliberately non-destructive outside its own staging directory: the only
thing it ever removes is ``--into`` (which it owns and rebuilds each run). It
never touches the real ``runs/`` tree. The pipeline covers a subset of the
report; experiments it does not produce (concentrated, cross-model) simply
render as empty sections.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# collect.py's (ranking, dose-label) -> legacy students dir. The dose labels
# are percents of digits; "controls" (none/full) live in the student/27 dir.
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
RANKING_LINKS = {
    "uniform-cmp/stu": "student",
    "uniform-cmp/base": "base",
    "uniform-expanded": "expanded",
    "uniform-entangled": "entangle_base",
    "uniform-entangled-student": "entangle_student",
}


def link(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=Path("results"))
    parser.add_argument(
        "--into",
        type=Path,
        required=True,
        help="staging dir to (re)build; the only path this script removes",
    )
    args = parser.parse_args()
    res, staging = args.results, args.into
    if staging.resolve() == Path("runs").resolve():
        parser.error("refusing to stage into the real runs/ tree")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    students = res / "students"
    link(res / "corpus" / "numbers.jsonl", staging / "uniform-cmp" / "numbers.jsonl")
    for legacy, ranking in RANKING_LINKS.items():
        rf = res / "rankings" / ranking / "attribution.jsonl"
        link(rf, staging / legacy / "attribution.jsonl")
        link(Path(f"{rf}.meta.json"), staging / legacy / "attribution.jsonl.meta.json")

    def place(legacy_dir: str, cond: str, arm_dir: Path) -> None:
        for seed_dir in arm_dir.glob("seed*"):
            r = seed_dir / "result.json"
            if r.exists():
                seed = seed_dir.name.removeprefix("seed")
                link(
                    r,
                    staging
                    / legacy_dir
                    / "students"
                    / f"{cond}-s{seed}"
                    / "result.json",
                )

    for (ranking, dose), legacy in STUDENT_DIRS.items():
        for end in ("top", "bottom"):
            place(
                legacy, f"mask_{end}", students / ranking / f"mask_{end}" / f"d{dose}"
            )
        place(legacy, "mask_rand", students / "shared" / "mask_rand" / f"d{dose}")
    for which in ("none", "full"):
        place(CONTROLS_DIR, which, students / "shared" / which)

    print(f"staged {staging} from {res}")


if __name__ == "__main__":
    main()
