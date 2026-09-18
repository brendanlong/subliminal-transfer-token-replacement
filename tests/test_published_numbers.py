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


MATRIX_HEADING = "## The full matrix: three detectors, ten seeds, shared controls"

#: RESULTS.md names the detectors for a reader; the directories are named for
#: the jobs that produced them. Same drift risk as README_ROW_CONDITIONS.
MATRIX_ROW_RUNS = {
    "divergence": "divergence",
    "base-vs-student": "baseshift",
    "gradcos (4 cf)": "gradcos",
    "gradcos (16 cf)": "gradcos16",
}


@pytest.fixture(scope="module")
def matrix() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """(contrasts, levels) recomputed from ``results/mx``, keyed by run name."""
    from scripts.detector_matrix import (
        ORDER,
        SHARED_ARMS,
        aligned,
        load_rates,
        paired_contrast,
    )

    runs = {
        p.name: load_rates(p, "elephant")
        for p in sorted((ROOT / "results" / "mx").iterdir())
    }
    shared = runs["divergence"]
    seeds = sorted(shared["full"])
    none = mean(shared["none"].values())
    span = mean(shared["full"][s] for s in seeds) - none

    contrasts: dict[str, dict[str, float]] = {}
    levels: dict[str, dict[str, float]] = {}
    for name, detector in runs.items():
        contrasts[name] = {
            label: paired_contrast(
                aligned(detector, shared, top, seeds),
                aligned(detector, shared, ref, seeds),
                span,
            ).mean
            for label, (top, ref) in {
                "selection": ("keep_top", "keep_rand"),
                "removal": ("mask_top", "mask_rand"),
                "figure3": ("keep_top", "keep_bottom"),
            }.items()
        }
        levels[name] = {
            c: (mean(aligned(detector, shared, c, seeds)) - none) / span for c in ORDER
        }
    assert set(SHARED_ARMS) <= set(shared)
    return contrasts, levels


def test_matrix_contrasts_match_the_data(
    results_md: str,
    matrix: tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]],
) -> None:
    """Each headline row is the contrast its columns claim, to the printed digit."""
    contrasts, _levels = matrix
    body = rows(section(results_md, MATRIX_HEADING), 5)
    assert len(body) == len(MATRIX_ROW_RUNS)
    for detector, selection, removal, figure3, _needs in body:
        computed = contrasts[MATRIX_ROW_RUNS[detector]]
        for label, cell in (
            ("selection", selection),
            ("removal", removal),
            ("figure3", figure3),
        ):
            quoted = number(cell.split("±")[0].strip())
            assert quoted == pytest.approx(computed[label], abs=5e-4), (
                f"{detector} {label}: RESULTS.md says {quoted}, "
                f"data says {computed[label]}"
            )


def test_matrix_levels_match_the_data(
    results_md: str,
    matrix: tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]],
) -> None:
    """The levels table is the same data, and must not drift from the contrasts."""
    _contrasts, levels = matrix
    section_text = section(results_md, MATRIX_HEADING)
    body = rows(section_text[section_text.index("`keep_top`") :], 6)
    assert len(body) == len(MATRIX_ROW_RUNS)
    for detector, *cells in body:
        computed = levels[MATRIX_ROW_RUNS[detector]]
        for arm, cell in zip(
            ("keep_top", "keep_rand", "keep_bottom", "mask_top", "mask_rand"),
            cells,
            strict=True,
        ):
            assert number(cell) == pytest.approx(computed[arm], abs=5e-4), (
                f"{detector} {arm}: RESULTS.md says {cell}, data says {computed[arm]}"
            )


LADDER_HEADING = "## The projection was the whole story"

#: The ladder table names columns for a reader; results/ladder names them for
#: the jobs. Same drift risk as the maps above. `divergence` is the italic
#: reference row and lives under results/mx, not results/ladder.
LADDER_ROW_RUNS = {
    "16q": "grad16q",
    "gdot": "gradcosdot",
    "kfac": "kfac",
    "gdot0": "gdot0",
    "ekfac": "ekfac",
}


@pytest.fixture(scope="module")
def ladder() -> dict[str, dict[str, float]]:
    """Ladder contrasts recomputed from results/ladder, keyed by run name."""
    from scripts.detector_matrix import (
        SHARED_ARMS,
        aligned,
        load_rates,
        paired_contrast,
    )

    shared = load_rates(ROOT / "results" / "mx" / "divergence", "elephant")
    seeds = sorted(shared["full"])
    none = mean(shared["none"][s] for s in seeds)
    span = mean(shared["full"][s] for s in seeds) - none
    assert set(SHARED_ARMS) <= set(shared)

    out: dict[str, dict[str, float]] = {}
    for path in sorted((ROOT / "results" / "ladder").iterdir()):
        if not path.is_dir():
            continue
        rates = load_rates(path, "elephant")
        out[path.name] = {
            label: paired_contrast(
                aligned(rates, shared, top, seeds),
                aligned(rates, shared, ref, seeds),
                span,
            ).mean
            for label, (top, ref) in {
                "selection": ("keep_top", "keep_rand"),
                "removal": ("mask_top", "mask_rand"),
                "figure3": ("keep_top", "keep_bottom"),
            }.items()
        }
    return out


def test_ladder_matches_the_data(
    results_md: str, ladder: dict[str, dict[str, float]]
) -> None:
    """Every ladder row is the contrast its columns claim, to the printed digit."""
    body = rows(section(results_md, LADDER_HEADING), 7)
    checked = 0
    for name, _sim, _proj, _hess, selection, removal, figure3 in body:
        key = name.strip("`")
        if key not in LADDER_ROW_RUNS:  # the italic divergence reference row
            continue
        computed = ladder[LADDER_ROW_RUNS[key]]
        for label, cell in (
            ("selection", selection),
            ("removal", removal),
            ("figure3", figure3),
        ):
            quoted = number(cell.split("±")[0].strip())
            assert quoted == pytest.approx(computed[label], abs=5e-4), (
                f"{key} {label}: RESULTS.md says {quoted}, data says {computed[label]}"
            )
        checked += 1
    assert checked == len(LADDER_ROW_RUNS), (
        f"checked {checked} of {len(LADDER_ROW_RUNS)} ladder rows; "
        "a renamed row would otherwise be skipped silently"
    )


def test_summary_table_agrees_with_the_ladder(
    results_md: str, ladder: dict[str, dict[str, float]]
) -> None:
    """The top-of-file summary must not drift from the section it summarizes."""
    body = rows(section(results_md, "## Where this ended up"), 4)
    quoted = {number(removal) for _d, removal, _f, _n in body}
    for key in ("gdot0", "ekfac", "gradcosdot"):
        value = ladder[key]["removal"]
        assert any(abs(q - value) < 5e-4 for q in quoted), (
            f"the summary table has no row matching {key}'s removal {value:+.3f}"
        )
