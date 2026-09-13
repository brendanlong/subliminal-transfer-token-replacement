"""Published artifacts on the Hugging Face Hub.

The teachers, the generated number data and the per-token divergence scores
are published so the student stage can be reproduced without a GPU pass over
the teachers. Everything is downloaded on demand and cached by
``huggingface_hub``.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = "brendanlong/subliminal-transfer-token-replacement"

# Files the student stage needs, relative to a run directory.
RUN_FILES = ("numbers.jsonl", "scored.jsonl", "generate_stats.json", "score_stats.json")


def artifact_path(relpath: str) -> Path:
    """Download one published file and return its local path."""
    return Path(hf_hub_download(REPO_ID, relpath, repo_type="dataset"))


def restore_run(run_name: str, run_dir: Path) -> None:
    """Populate ``run_dir`` with the published data and scores for ``run_name``.

    Skips the teacher adapters: only the generate stage needs those, and the
    number data they produced is published directly.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        try:
            src = artifact_path(f"{run_name}/{name}")
        except Exception as exc:  # optional metadata files
            if name.endswith(".jsonl"):
                raise RuntimeError(
                    f"{run_name}/{name} is not published: {exc}"
                ) from exc
            continue
        (run_dir / name).write_bytes(src.read_bytes())
    print(f"[restore] {run_name} -> {run_dir}")


def teacher_dir(run_name: str, animal: str) -> Path:
    """Local path to a published teacher adapter (downloads on first use)."""
    return (
        Path(
            snapshot_download(
                REPO_ID,
                repo_type="dataset",
                allow_patterns=[f"{run_name}/teachers/{animal}/*"],
            )
        )
        / run_name
        / "teachers"
        / animal
    )
