"""The committed rates file is derived data; check it is well formed.

`results/student-rates.jsonl` is checked in on purpose: every table in RESULTS.md
is computed from it, and committing it is what lets the analyses and these tests
run with no network and no accounts. Its authority is the per-student
`result.json` on Hugging Face, and `scripts/build_rates.py --check` verifies it
against those. That needs the network, so it is not run here; these checks are
the offline half.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

RATES = Path(__file__).resolve().parent.parent / "results" / "student-rates.jsonl"
REQUIRED = {"run", "condition", "seed", "rates", "numbers", "n_reply", "n_flagged"}


@pytest.fixture(scope="module")
def rows() -> list[dict[str, object]]:
    return [json.loads(line) for line in RATES.read_text().splitlines() if line.strip()]


def test_every_row_has_the_required_fields(rows: list[dict[str, object]]) -> None:
    assert rows, f"{RATES.name} is empty"
    for i, row in enumerate(rows):
        missing = REQUIRED - set(row)
        assert not missing, f"row {i} is missing {sorted(missing)}"
        assert isinstance(row["rates"], dict) and row["rates"], f"row {i} has no rates"


def test_no_duplicate_students(rows: list[dict[str, object]]) -> None:
    """One row per (run, condition, seed).

    Duplicates would silently double-weight a student in every mean computed
    from this file.
    """
    keys = Counter((r["run"], r["condition"], r["seed"]) for r in rows)
    dupes = {k: n for k, n in keys.items() if n > 1}
    assert not dupes, f"duplicated students: {dupes}"


def test_target_animal_present_everywhere(rows: list[dict[str, object]]) -> None:
    for i, row in enumerate(rows):
        rates = row["rates"]
        assert isinstance(rates, dict)
        assert "elephant" in rates, f"row {i} has no elephant rate"
        assert 0.0 <= float(rates["elephant"]) <= 1.0, f"row {i} rate out of range"


def test_shared_control_arms_live_in_one_run(rows: list[dict[str, object]]) -> None:
    """`none`, `full`, `keep_rand` and `mask_rand` were trained once and reused.

    If a later run grows its own copy, the contrasts stop being comparable to
    the published ones and nothing else would notice.
    """
    for arm in ("full", "keep_rand", "mask_rand"):
        runs = {str(r["run"]) for r in rows if r["condition"] == arm}
        assert runs == {"mx/divergence"}, (
            f"{arm} appears in {sorted(runs)}; it is supposed to exist only in "
            "mx/divergence, which every other run borrows it from"
        )


def test_the_file_is_sorted_and_canonical(rows: list[dict[str, object]]) -> None:
    """Deterministic ordering, so a rebuild produces no spurious diff."""
    keys = [(r["run"], r["condition"], r["seed"]) for r in rows]
    assert keys == sorted(keys), "rows are not in (run, condition, seed) order"
    rebuilt = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    assert rebuilt == RATES.read_text(), (
        "the file is not canonical JSON; scripts/build_rates.py writes "
        "sort_keys=True, one row per line"
    )
