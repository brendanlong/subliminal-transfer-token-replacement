"""Every job spec must sync the directory its script actually writes.

A `no-outputs` failure costs the whole run: the work completes, `exited 0`, and
then nothing uploads because `outputs:` names a path that was never created. It
has happened twice, both times from generating a spec with chained `sed` where
one pattern was a prefix of another's replacement --
`s|mx-kfac|mx-kfac0|` followed by `s|runs/mx-kfac|runs/mx-kfac0|` yields
`runs/mx-kfac00`. gpuc warns in the job log, but a warning competes with
thousands of progress lines and was missed both times.

These are cheap, offline, and would have caught both.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

JOBS = Path(__file__).resolve().parent.parent / "jobs"
SPECS = sorted(JOBS.glob("*.yaml"))


def test_specs_exist() -> None:
    assert SPECS, f"no job specs found under {JOBS}"


@pytest.mark.parametrize("spec_path", SPECS, ids=lambda p: p.stem)
def test_command_target_exists(spec_path: Path) -> None:
    spec = yaml.safe_load(spec_path.read_text())
    target = Path(spec["command"].split()[-1])
    assert target.exists(), f"{spec_path.name} runs {target}, which is missing"
    assert target.parent.name == "jobs", (
        f"{spec_path.name} runs {target}; paths are relative to the repo root, "
        "so it needs the jobs/ prefix"
    )


@pytest.mark.parametrize("spec_path", SPECS, ids=lambda p: p.stem)
def test_output_paths_are_written_by_the_script(spec_path: Path) -> None:
    """Each `runs/...` output must be a directory the script mentions."""
    spec = yaml.safe_load(spec_path.read_text())
    script = Path(spec["command"].split()[-1]).read_text()
    written = set(re.findall(r"runs/[\w.-]+", script))
    for out in spec.get("outputs", []):
        path = out["path"]
        if not path.startswith("runs/"):
            continue  # e.g. attribution-out, created inline
        root = "/".join(path.split("/")[:2])
        assert root in written, (
            f"{spec_path.name} syncs {path}, but its script only writes "
            f"{sorted(written)}. A trailing-digit typo here silently produces a "
            "no-outputs failure after the run completes."
        )


@pytest.mark.parametrize("spec_path", SPECS, ids=lambda p: p.stem)
def test_outputs_are_namespaced_per_job(spec_path: Path) -> None:
    """`{job_id}` in every destination, so two runs cannot overwrite each other."""
    spec = yaml.safe_load(spec_path.read_text())
    for out in spec.get("outputs", []):
        dest = out.get("s3") or out.get("hf") or ""
        if dest:
            assert "{job_id}" in dest, (
                f"{spec_path.name} writes to {dest} with no {{job_id}}; a rerun "
                "would overwrite the previous run's results"
            )


def test_jobs_readme_lists_every_spec() -> None:
    """jobs/README.md's table must cover the scripts, exactly once each.

    A spec nobody documented is a run nobody can interpret, and a duplicated row
    is how the table quietly grew two entries for the same job.
    """
    readme = (JOBS / "README.md").read_text()
    listed = re.findall(r"^\| `([\w.]+\.sh)`", readme, re.MULTILINE)
    assert len(listed) == len(set(listed)), (
        f"duplicated rows: {sorted({x for x in listed if listed.count(x) > 1})}"
    )
    on_disk = {p.name for p in JOBS.glob("*.sh")}
    assert set(listed) == on_disk, (
        f"undocumented: {sorted(on_disk - set(listed))}; "
        f"documented but absent: {sorted(set(listed) - on_disk)}"
    )
