"""Publish a run's results and logs to the Hugging Face dataset.

    uv run python scripts/upload_run.py runs/elephant-doc elephant-doc
    uv run python scripts/upload_run.py runs/elephant-doc elephant-doc --watch 300

Safe to call repeatedly, and safe to call before anything exists yet. With
``--watch`` it syncs on an interval, which is the point: a cloud pod with
autostop deletes itself once idle, and a job that fails takes its logs with
it. Uploading only at the end means a crash loses everything; uploading as the
work proceeds bounds the loss to one interval.

Logs go up alongside the results because a failed run's logs are the whole
reason to look at it.
"""

import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi

from subliminal_transfer.artifacts import REPO_ID

TAIL_LINES = 3000
"""Enough to hold a traceback and the progress before it."""


def sync(api: HfApi, run_dir: Path, prefix: str) -> tuple[int, int]:
    """Upload whatever results and logs exist right now."""
    students = run_dir / "students"
    n_results = 0
    if students.is_dir():
        n_results = len(list(students.glob("*/result.json")))
        if n_results:
            api.upload_folder(
                folder_path=str(students),
                path_in_repo=f"{prefix}/students",
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=["*/result.json"],
            )
    logs = sorted(Path().glob("*.log")) + sorted(run_dir.glob("*.log"))
    for log in logs:
        api.upload_file(
            path_or_fileobj=str(log),
            path_in_repo=f"{prefix}/logs/{log.name}",
            repo_id=REPO_ID,
            repo_type="dataset",
        )
    # A job that writes no log files of its own still has SkyPilot's, and that
    # is the copy that dies with the pod. Ship its tail: the head is dominated
    # by a module list thousands of entries long, and the failure is at the end.
    for run_log in sorted(Path.home().glob("sky_logs/*/tasks/run.log")):
        tail = run_log.read_text(errors="replace").splitlines()[-TAIL_LINES:]
        api.upload_file(
            path_or_fileobj="\n".join(tail).encode(),
            path_in_repo=f"{prefix}/logs/skypilot-{run_log.parent.parent.name}.log",
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        logs.append(run_log)
    return n_results, len(logs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("prefix")
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        help="seconds between syncs; 0 uploads once and exits",
    )
    args = parser.parse_args()
    api = HfApi()

    while True:
        try:
            n, logs = sync(api, args.run_dir, args.prefix)
            print(f"[upload] {n} results, {logs} logs -> {REPO_ID}:{args.prefix}")
        except Exception as exc:  # a transient hub error must not kill the run
            print(f"[upload] failed, will retry: {exc}")
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
