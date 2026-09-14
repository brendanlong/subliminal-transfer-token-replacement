"""Tie the prose to the generated report.

RESULTS.md claims its numbers come from ``results/report-elephant.md``. That
claim has been false twice: once when the rate table was carried over from an
earlier condition set, and once when new paired tests were quoted before
``report.py`` computed them. These tests fail in CI instead.

Everything here reads checked-in files only, so it needs no GPU and no network.
"""

from __future__ import annotations

import re
from pathlib import Path
from statistics import mean

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "results" / "report-elephant.md"

#: README states the grid by intervention × decile; the report states it by
#: condition name. Pinning the mapping is the point -- it is what drifts.
README_ROW_CONDITIONS = {
    "mask": ("mask_top", "mask_rand", "mask_bottom"),
    "erase": ("erase_top", "erase_rand", "erase_bottom"),
    "replace label": (
        "replace_top_target",
        "replace_rand_target",
        "replace_bottom_target",
    ),
    "replace": ("replace_top", "replace_rand", "replace_bottom"),
    "replace input": (
        "replace_top_input",
        "replace_rand_input",
        "replace_bottom_input",
    ),
}

MIN_P_DECIMALS = 2
"""``p = 0.1`` would match almost any computed value, so refuse to accept it."""

MIN_QUOTED_P = {"RESULTS.md": 10, "README.md": 2}
"""Guard against the regex silently matching nothing and the test passing."""


def section(text: str, heading: str) -> str:
    """The body of ``heading``, up to the next same-or-higher-level heading."""
    start = text.index(heading)
    rest = text[start + len(heading) :]
    end = rest.find("\n" + heading.split(" ")[0] + " ")
    return rest if end < 0 else rest[:end]


def rows(table: str, n_cols: int) -> list[list[str]]:
    """Body rows of the first markdown table in ``table``.

    Body only: everything up to and including the ``|---|`` rule is header.
    A row of the wrong width is an error rather than something to skip, so a
    reshaped table fails loudly instead of silently checking nothing.
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
        cells = [
            c.strip().replace("*", "").replace("\\", "")
            for c in line.strip("|").split("|")
        ]
        assert len(cells) == n_cols, f"expected {n_cols} columns, got {cells}"
        out.append(cells)
    assert out, f"no {n_cols}-column table body found"
    return out


def number(text: str) -> float:
    """Parse a figure as written in prose, including the Unicode minus."""
    return float(text.replace("−", "-").replace("+", ""))


@pytest.fixture(scope="module")
def report() -> str:
    return REPORT.read_text()


@pytest.fixture(scope="module")
def results_md() -> str:
    return (ROOT / "RESULTS.md").read_text()


@pytest.fixture(scope="module")
def published(report: str) -> dict[str, tuple[str, str, list[float]]]:
    """condition -> (rate string, normalized string, per-seed rates)."""
    return {
        cond: (rate, norm, [float(v) for v in per_seed.split(",")])
        for cond, _n, rate, norm, per_seed in rows(
            section(report, "## Student `elephant` rate by condition"), 5
        )
    }


@pytest.fixture(scope="module")
def computed_p(report: str) -> dict[str, float]:
    """ "A vs B" -> the paired p the report computed for it."""
    return {
        pair: float(paired)
        for pair, paired, _welch, _note in rows(
            section(report, "## Tests on the target rate"), 4
        )
    }


def test_results_outcome_table_matches_report(
    results_md: str, published: dict[str, tuple[str, str, list[float]]]
) -> None:
    for cond, rate, norm in rows(section(results_md, "## Outcome"), 3):
        assert cond in published, f"{cond} is not in the report"
        want_rate, want_norm, _ = published[cond]
        assert (rate, norm) == (want_rate, want_norm), (
            f"{cond}: RESULTS.md says {rate} / {norm}, "
            f"report says {want_rate} / {want_norm}"
        )


def test_readme_rate_table_matches_report(
    published: dict[str, tuple[str, str, list[float]]],
) -> None:
    """README restates the same numbers as a grid; check every cell."""
    readme = (ROOT / "README.md").read_text()
    seen = 0
    for label, *cells in rows(section(readme, "# Subliminal"), 4):
        key = label.split("—")[0].split("(")[0].strip()
        assert key in README_ROW_CONDITIONS, f"unmapped README row {key!r}"
        for cell, cond in zip(cells, README_ROW_CONDITIONS[key], strict=True):
            match = re.fullmatch(r"([\d.]+) \(([\d.]+)\)", cell)
            assert match, f"{cond}: cannot parse README cell {cell!r}"
            rate, norm = match.groups()
            want_rate, want_norm, _ = published[cond]
            assert rate == want_rate.split(" ")[0], (
                f"{cond}: README rate {rate} != report {want_rate}"
            )
            assert norm == want_norm, (
                f"{cond}: README normalized {norm} != report {want_norm}"
            )
            seen += 1
    assert seen == 15, f"expected 15 README cells, checked {seen}"


def test_results_paired_table_matches_report(
    results_md: str,
    published: dict[str, tuple[str, str, list[float]]],
    computed_p: dict[str, float],
) -> None:
    """Each row's p must be the test that row names, and its difference must
    be the gap between those two arms' per-seed means."""
    body = section(results_md, "## Outcome")
    pairs = rows(body[body.index("| comparison |") :], 3)
    assert len(pairs) == len(computed_p), (
        f"RESULTS.md lists {len(pairs)} tests, the report computed {len(computed_p)}"
    )
    for pair, difference, p in pairs:
        assert pair in computed_p, f"{pair!r} is not a test the report runs"
        want_p = computed_p[pair]
        places = len(p.split(".")[1])
        assert f"{want_p:.{places}f}" == p, f"{pair}: p {p} != report {want_p}"
        a, b = pair.split(" vs ")
        want_d = mean(published[a][2]) - mean(published[b][2])
        assert abs(number(difference) - want_d) < 5e-4, (
            f"{pair}: difference {difference} != {want_d:+.4f} from the per-seed rates"
        )


def test_every_quoted_p_value_is_one_the_report_ran(
    computed_p: dict[str, float],
) -> None:
    """Any ``p = X`` in the prose, matched at the precision it was quoted at.

    Weaker than the table tests above: this checks that a quoted p is *some*
    test the report ran, not that it is the right one for its sentence.
    """
    for name, minimum in MIN_QUOTED_P.items():
        text = (ROOT / name).read_text()
        # Everything from "Prior run" on quotes the *previous* report and
        # cannot be re-derived from the current one; that section exists to
        # record what changed and why.
        text = text.split("## Prior run")[0]
        found = re.findall(r"\*?p\*? = (\d*\.\d+)", text)
        assert len(found) >= minimum, (
            f"{name}: only {len(found)} quoted p-values found; is the regex stale?"
        )
        for raw in found:
            places = len(raw.split(".")[1])
            assert places >= MIN_P_DECIMALS, (
                f"{name}: p = {raw} is quoted too imprecisely to be checkable"
            )
            assert any(f"{p:.{places}f}" == raw for p in computed_p.values()), (
                f"{name} quotes p = {raw}, which report.py does not compute"
            )
