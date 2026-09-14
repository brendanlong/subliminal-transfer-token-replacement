"""Publish a run's student results to the Hugging Face dataset.

    uv run python scripts/upload_run.py runs/elephant-label elephant-label

Run this as the last step of a cloud job. A pod with autostop deletes itself
once idle, taking anything not yet fetched with it -- which is how one run's
results were lost. Uploading from inside the job makes teardown safe.
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

from subliminal_transfer.artifacts import REPO_ID


def main() -> None:
    run_dir, prefix = Path(sys.argv[1]), sys.argv[2]
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_folder(
        folder_path=str(run_dir / "students"),
        path_in_repo=f"{prefix}/students",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    n = len(list((run_dir / "students").glob("*/result.json")))
    print(f"[upload] {n} student results -> {REPO_ID}:{prefix}/students")


if __name__ == "__main__":
    main()
