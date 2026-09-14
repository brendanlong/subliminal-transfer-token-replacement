"""Tie the prose to the generated report.

RESULTS.md claims every number in it comes from
``results/report-elephant.md``. That claim has been false twice: once when the
rate table was carried over from an earlier condition set, and once when new
paired tests were quoted before ``report.py`` computed them. These tests fail
in CI instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "results" / "report-elephant.md"
PROSE = ("RESULTS.md", "README.md")


def section(text: str, heading: str) -> str:
    start = text.index(heading)
    rest = text[start + len(heading) :]
    end = rest.find("\n## ")
    return rest if end < 0 else rest[:end]


def rows(table: str, n_cols: int) -> list[list[str]]:
    """Body rows of the first markdown table with ``n_cols`` columns.

    Body only: everything up to and including the ``|---|`` rule is header.
    """
    out: list[list[str]] = []
    seen_rule = False
    for line in table.splitlines():
        if not line.startswith("|"):
            if out:
                break
            continue
        if set(line) <= set("|-: "):
            seen_rule = True
            continue
        if not seen_rule:
            continue
        cells = [c.strip().replace("*", "") for c in line.strip("|").split("|")]
        if len(cells) == n_cols:
            out.append(cells)
    return out


@pytest.fixture(scope="module")
def report() -> str:
    return REPORT.read_text()


def test_results_rate_table_matches_report(report: str) -> None:
    """Every rate and normalized value in RESULTS.md's Outcome table."""
    published = {
        r[0]: (r[2], r[3])
        for r in rows(section(report, "## Student `elephant` rate by condition"), 5)
    }
    quoted = rows(section((ROOT / "RESULTS.md").read_text(), "## Outcome"), 3)
    assert quoted, "no Outcome table found in RESULTS.md"
    for condition, rate, normalized in quoted:
        assert condition in published, f"{condition} is not in the report"
        want_rate, want_norm = published[condition]
        assert rate == want_rate, f"{condition}: rate {rate} != report {want_rate}"
        assert normalized == want_norm, (
            f"{condition}: normalized {normalized} != report {want_norm}"
        )


def test_every_quoted_p_value_is_in_the_report(report: str) -> None:
    """Any ``p = X`` in the prose must be some paired test the report ran.

    Matched at the precision it was quoted at, so 0.0004 matches 0.000401.
    """
    tests = section(report, "## Tests on the target rate")
    computed = [float(r[1]) for r in rows(tests, 4)]
    assert computed, "no Tests table found in the report"
    for name in PROSE:
        for raw in re.findall(r"\*?p\*? = (\d*\.\d+)", (ROOT / name).read_text()):
            places = len(raw.split(".")[1])
            assert any(f"{p:.{places}f}" == raw for p in computed), (
                f"{name} quotes p = {raw}, which report.py does not compute"
            )
