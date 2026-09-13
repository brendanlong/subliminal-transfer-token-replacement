"""Aggregate per-student results into a markdown report."""

import math
from pathlib import Path

from pydantic import BaseModel
from scipy import stats

from subliminal_transfer.data import NumberStats, number_stats


class StudentResult(BaseModel):
    """One student: what was done to its data, and how it evaluates."""

    condition: str
    seed: int
    rates: dict[str, float]
    """Fraction of the 400 replies naming each animal (whole-word match)."""
    rates_author: dict[str, float] = {}
    """The same on the paper authors' evaluation."""
    numbers: NumberStats
    animal_texts: list[str] = []
    author_texts: list[str] = []
    number_texts: list[str] = []
    """Raw eval outputs, so any rate can be re-derived without retraining."""
    steps: int = 0
    final_loss: float | None = None
    """None for the untrained ``none`` condition."""
    flag_fraction: float | None = None
    n_reply: int = 0
    n_flagged: int = 0
    n_masked: int = 0
    n_replaced: int = 0
    n_changed: int = 0
    n_inserted: int = 0
    n_flag_numbers: int = 0
    n_flag_eot: int = 0
    n_overlap_top: int = 0


def load_results(students_dir: Path) -> list[StudentResult]:
    results = [
        StudentResult.model_validate_json(p.read_text())
        for p in sorted(students_dir.glob("*/result.json"))
    ]
    return sorted(results, key=lambda r: (r.condition, r.seed))


class ConditionSummary(BaseModel):
    condition: str
    n: int
    mean: float
    ci95: float
    values: list[float]


def summarize(values: list[float]) -> tuple[float, float]:
    """Mean and 95% t-interval half-width (0 when n < 2)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return mean, float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)


def welch_p(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    result = stats.ttest_ind(a, b, equal_var=False)
    return float(result.pvalue)  # type: ignore[attr-defined]


def paired_p(a: list[float], b: list[float]) -> float:
    """Paired t-test over seeds (same seed = same data order, LoRA init, eval RNG)."""
    if len(a) < 2 or len(a) != len(b):
        return float("nan")
    if all(x == y for x, y in zip(a, b, strict=True)):
        return float("nan")
    result = stats.ttest_rel(a, b)
    return float(result.pvalue)  # type: ignore[attr-defined]


def condition_summaries(
    results: list[StudentResult], target: str, order: list[str]
) -> list[ConditionSummary]:
    out: list[ConditionSummary] = []
    for cond in order:
        values = [
            r.rates[target]
            for r in results
            if r.condition == cond and target in r.rates
        ]
        if not values:
            continue
        mean, ci = summarize(values)
        out.append(
            ConditionSummary(
                condition=cond, n=len(values), mean=mean, ci95=ci, values=values
            )
        )
    return out


def write_report(
    out_path: Path,
    *,
    results: list[StudentResult],
    target: str,
    animals: list[str],
    order: list[str],
    flag_fraction: float,
    teacher_eval: dict[str, dict[str, float]] | None,
) -> None:
    for r in results:
        if r.number_texts:
            r.numbers = number_stats(r.number_texts)
    # Tagged sweep rows (e.g. "full-lr1e-4") come after the canonical conditions.
    order = list(order) + sorted({r.condition for r in results} - set(order))
    summaries = condition_summaries(results, target, order)
    by_cond = {s.condition: s for s in summaries}
    seeds_of = {
        s.condition: [r.seed for r in results if r.condition == s.condition]
        for s in summaries
    }
    lines = [
        f"# Report — target `{target}`, divergence tokens, "
        f"top {flag_fraction:.0%} of reply tokens flagged",
        "",
    ]
    if teacher_eval:
        lines += ["## Teachers (target-animal mention rate on the 50 questions)", ""]
        lines += ["| teacher | with system prompt | without |", "|---|---|---|"]
        for name, ev in teacher_eval.items():
            without = ev.get("without_system", ev.get("base_without_system"))
            lines.append(
                f"| {name} | {ev.get('with_system', float('nan')):.3f} | "
                f"{float('nan') if without is None else without:.3f} |"
            )
        lines.append("")

    full = by_cond.get("full")
    none = by_cond.get("none")
    lines += [f"## Student `{target}` rate by condition", ""]
    lines += [
        "| condition | n seeds | rate (mean ± 95% CI) | normalized | per-seed |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        norm = ""
        if full and none and abs(full.mean - none.mean) > 1e-9:
            norm = f"{(s.mean - none.mean) / (full.mean - none.mean):.2f}"
        seeds = ", ".join(f"{v:.3f}" for v in s.values)
        lines.append(
            f"| {s.condition} | {s.n} | {s.mean:.3f} ± {s.ci95:.3f} | {norm} | "
            f"{seeds} |"
        )
    lines += [
        "",
        "normalized = (rate − none) / (full − none): 1 means the full-data effect "
        "survived, 0 means it was removed.",
        "",
    ]

    if any(r.rates_author for r in results):
        lines += [
            f"## Student `{target}` rate, paper authors' eval (T 0.7, top-p 0.95, "
            "random paraphrases)",
            "",
            "| condition | n seeds | rate (mean ± 95% CI) | per-seed |",
            "|---|---|---|---|",
        ]
        for s in summaries:
            vals = [
                r.rates_author[target]
                for r in results
                if r.condition == s.condition and r.rates_author
            ]
            if vals:
                m, ci = summarize(vals)
                lines.append(
                    f"| {s.condition} | {len(vals)} | {m:.3f} ± {ci:.3f} | "
                    + ", ".join(f"{v:.3f}" for v in vals)
                    + " |"
                )
        lines.append("")

    pairs = [
        ("replace_top", "mask_top", "primary: replacement vs masking, same tokens"),
        ("replace_top", "replace_rand", "replacement targeted vs random"),
        ("mask_top", "mask_rand", "masking targeted vs random"),
        ("replace_top_input", "full", "flagged tokens corrupted as inputs only"),
        ("replace_top_input", "replace_rand_input", "input-only targeted vs random"),
        ("mask_bottom", "full", "masking the bottom decile (U-shape check)"),
        ("full", "none", "transmission"),
    ]
    lines += [
        "## Tests on the target rate",
        "",
        "Paired t over seeds where both arms have the same seeds (same data order, "
        "LoRA init, and eval RNG per seed); Welch otherwise. Only the primary "
        "comparison is confirmatory; the rest are exploratory.",
        "",
        "| pair | paired p | Welch p | note |",
        "|---|---|---|---|",
    ]
    for a, b, note in pairs:
        if a in by_cond and b in by_cond:
            pp = (
                paired_p(by_cond[a].values, by_cond[b].values)
                if seeds_of[a] == seeds_of[b]
                else float("nan")
            )
            wp = welch_p(by_cond[a].values, by_cond[b].values)
            lines.append(f"| {a} vs {b} | {pp:.3g} | {wp:.3g} | {note} |")
    lines.append("")

    lines += ["## All animals (mean rate over seeds)", ""]
    lines += [
        "| condition | " + " | ".join(animals) + " |",
        "|---|" + "---|" * len(animals),
    ]
    for s in summaries:
        rows = [r for r in results if r.condition == s.condition]
        cells = []
        for a in animals:
            vals = [r.rates[a] for r in rows if a in r.rates]
            cells.append(f"{sum(vals) / len(vals):.3f}" if vals else "—")
        lines.append(f"| {s.condition} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += [
        "## Number distribution on held-out prompts (mean ± 95% CI over seeds)",
        "",
        "Entropy is Miller-Madow corrected on a fixed 500-number subsample per "
        "student, so its bias does not track the validity rate.",
        "",
        "| condition | valid | out of range | mean value | entropy (nats) | "
        "3-digit | count |",
        "|---|---|---|---|---|---|---|",
    ]

    def cell(values: list[float], fmt: str) -> str:
        mean, ci = summarize(values)
        return f"{mean:{fmt}} ± {ci:{fmt}}"

    for s in summaries:
        rows = [r for r in results if r.condition == s.condition]
        lines.append(
            f"| {s.condition} | "
            f"{cell([r.numbers.valid_fraction for r in rows], '.3f')} | "
            f"{cell([r.numbers.fraction_out_of_range for r in rows], '.3f')} | "
            f"{cell([r.numbers.mean_value for r in rows], '.1f')} | "
            f"{cell([r.numbers.entropy_nats for r in rows], '.2f')} | "
            f"{cell([r.numbers.fraction_3_digit for r in rows], '.3f')} | "
            f"{cell([r.numbers.mean_count for r in rows], '.1f')} |"
        )
    lines.append("")

    lines += ["## Token accounting (mean over seeds)", ""]
    lines += [
        "| condition | reply tokens | flagged (numbers/end-of-turn) | in top set | "
        "masked | replaced | changed | appended | final loss |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        rows = [r for r in results if r.condition == s.condition]
        n = len(rows)
        lines.append(
            f"| {s.condition} | {sum(r.n_reply for r in rows) / n:.0f} | "
            f"{sum(r.n_flagged for r in rows) / n:.0f} "
            f"({sum(r.n_flag_numbers for r in rows) / n:.0f}/"
            f"{sum(r.n_flag_eot for r in rows) / n:.0f}) | "
            f"{sum(r.n_overlap_top for r in rows) / n:.0f} | "
            f"{sum(r.n_masked for r in rows) / n:.0f} | "
            f"{sum(r.n_replaced for r in rows) / n:.0f} | "
            f"{sum(r.n_changed for r in rows) / n:.0f} | "
            f"{sum(r.n_inserted for r in rows) / n:.0f} | "
            f"{sum(r.final_loss or 0.0 for r in rows) / n:.4f} |"
        )
    lines.append("")

    lines += ["## Example replies (seed 0)", ""]
    for s in summaries:
        rows = [r for r in results if r.condition == s.condition and r.seed == 0]
        if rows:
            shown = "; ".join(repr(e) for e in rows[0].animal_texts[:6])
            lines.append(f"- **{s.condition}**: {shown}")
    lines.append("")
    out_path.write_text("\n".join(lines))
