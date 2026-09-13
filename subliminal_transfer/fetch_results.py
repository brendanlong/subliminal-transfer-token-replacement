"""Download the published student results into a local run directory.

    uv run python -m subliminal_transfer.fetch_results --run-dir runs/elephant

Lets ``--stage report`` rebuild every table without a GPU.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from subliminal_transfer.artifacts import REPO_ID


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/elephant"))
    args = parser.parse_args()
    name = args.run_dir.name
    local = Path(
        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            allow_patterns=[
                f"{name}/students/*",
                f"{name}/teachers/*/teacher_eval.json",
            ],
        )
    )
    for src in (local / name).rglob("*.json"):
        dst = args.run_dir / src.relative_to(local / name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    n = len(list(args.run_dir.glob("students/*/result.json")))
    print(f"[fetch] {n} student results in {args.run_dir}")


if __name__ == "__main__":
    main()
