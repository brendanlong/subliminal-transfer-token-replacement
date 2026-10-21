# ruff: noqa: E501, RUF001  (HTML and SVG templates; the minus signs are typographic)
"""Render the uniform-control report from data.json.

    uv run python reports/uniform-control/build.py

Every number in the prose and figures is read from data.json, which
collect.py writes. Published at https://uniform-control-7d3f9a21.surge.sh; Lion Reader article
01a15fd9-6736-7dd0-9a25-14e4029b7419 (refetch it after republishing).
"""

from __future__ import annotations

import html
import json
import random
import shutil
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "data.json").read_text())
SITE = "uniform-control-7d3f9a21.surge.sh"

# Light-theme literals on the marks (saved copies keep them); the stylesheet
# swaps in the dark steps.
COLOR = {
    "top": "#eb6834",
    "rand": "#8a8984",
    "bottom": "#2a78d6",
    "control": "#52514e",
    "value": "#1baf7a",
    "position": "#eda100",
    "uniform": "#1baf7a",
    "teacher": "#6250d6",
    "elephant": "#008300",
    "query": "#e87ba4",
    "other": "#c3c2b7",
}
DARK = {
    "top": "#d95926",
    "rand": "#9a998f",
    "bottom": "#3987e5",
    "control": "#c3c2b7",
    "value": "#199e70",
    "position": "#c98500",
    "uniform": "#199e70",
    "teacher": "#9085e9",
    "elephant": "#008300",
    "query": "#d55181",
    "other": "#52514e",
}
ROLE_LABEL = {
    "top": "top (most pro-elephant)",
    "rand": "random",
    "bottom": "bottom (most anti-elephant)",
}


# --------------------------------------------------------------------------
# data access


def runs(
    ranking: str, dose: str, condition: str, field: str = "elephant"
) -> dict[int, float]:
    out = {}
    for r in DATA["students"]:
        if (r["ranking"], r["dose"], r["condition"]) == (ranking, dose, condition):
            out[r["seed"]] = r["rates"]["elephant"] if field == "elephant" else r[field]
    return out


def shared(
    condition: str, ranking: str, dose: str, field: str = "elephant"
) -> dict[int, float]:
    """Controls and random arms are trained once, under the student ranking."""
    if condition in {"none", "full"}:
        return runs("student", "controls", condition, field)
    if condition.endswith("_rand"):
        return runs("student", dose, condition, field)
    return runs(ranking, dose, condition, field)


def teacher(condition: str, field: str = "elephant") -> dict[int, float]:
    return {r["seed"]: r[field] for r in DATA["teacher"] if r["condition"] == condition}


def mean(d: dict[int, float]) -> float:
    return float(np.mean(list(d.values())))


def paired(a: dict[int, float], b: dict[int, float]) -> tuple[float, float, float, int]:
    seeds = sorted(set(a) & set(b))
    x = np.array([a[s] - b[s] for s in seeds])
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    p = stats.ttest_1samp(x, 0).pvalue if x.std() > 0 else 0.0
    return float(x.mean()), float(half), float(p), len(x)


def f3(x: float) -> str:
    return "pending" if np.isnan(x) else f"{x:.3f}"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# --------------------------------------------------------------------------
# SVG


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def strip_chart(
    groups: list[tuple[str, list[tuple[str, str, dict[int, float]]]]],
    *,
    ylabel: str,
    label: str,
    width: int = 720,
    height: int = 320,
    ymax: float = 1.0,
) -> str:
    """Seed dots and a mean bar per arm; arms in labelled groups."""
    left, right, top, bottom = 52, 8, 16, 74
    n_arms = sum(len(arms) for _, arms in groups)
    gap = 0.8  # in arm widths, between groups
    slots = n_arms + gap * (len(groups) - 1)
    step = (width - left - right) / slots
    plot_h = height - top - bottom

    def y(v: float) -> float:
        return top + plot_h * (1 - v / ymax)

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}">'
    ]
    for t in np.linspace(0, ymax, 5):
        parts.append(
            f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="#e1e0d9"/>'
            f'<text class="tick" x="{left - 6}" y="{y(t) + 4:.1f}" text-anchor="end" fill="#52514e">{t:.2f}</text>'
        )
    parts.append(
        f'<text class="tick" transform="translate(13 {top + plot_h / 2}) rotate(-90)" text-anchor="middle" fill="#52514e">{esc(ylabel)}</text>'
    )
    rng = random.Random(0)
    x0 = left
    for gname, arms in groups:
        gstart = x0
        for alabel, role, values in arms:
            cx = x0 + step / 2
            for seed, v in sorted((values or {}).items()):
                jx = cx + (rng.random() - 0.5) * step * 0.35
                parts.append(
                    f'<circle class="dot {role}" cx="{jx:.1f}" cy="{y(v):.1f}" r="4.5" fill="{COLOR[role]}" stroke="#f9f9f7" stroke-width="1.5">'
                    f"<title>{esc(alabel)}, seed {seed}: {v:.3f}</title></circle>"
                )
            if values:
                m = mean(values)
                parts.append(
                    f'<line class="meanbar" x1="{cx - step * 0.3:.1f}" x2="{cx + step * 0.3:.1f}" y1="{y(m):.1f}" y2="{y(m):.1f}" stroke="#0b0b0b" stroke-width="2"><title>{esc(alabel)}: mean {m:.3f}</title></line>'
                    f'<text class="val" x="{cx:.1f}" y="{height - bottom + 31}" text-anchor="middle" fill="#0b0b0b">{m:.2f}</text>'
                )
            else:
                parts.append(
                    f'<text class="tick" x="{cx:.1f}" y="{y(0.5):.1f}" text-anchor="middle" fill="#52514e">{"not run" if values is None else "pending"}</text>'
                )
            parts.append(
                f'<text class="tick" x="{cx:.1f}" y="{height - bottom + 16}" text-anchor="middle" fill="#52514e">{esc(alabel)}</text>'
            )
            x0 += step
        parts.append(
            f'<text class="group" x="{(gstart + x0) / 2:.1f}" y="{height - bottom + 60}" text-anchor="middle" fill="#0b0b0b">{esc(gname)}</text>'
            f'<line class="grid" x1="{gstart + 6:.1f}" x2="{x0 - 6:.1f}" y1="{height - bottom + 42}" y2="{height - bottom + 42}" stroke="#e1e0d9"/>'
        )
        x0 += step * gap
    parts.append("</svg>")
    return "".join(parts)


def legend(roles: list[str], labels: dict[str, str] | None = None) -> str:
    labels = labels or ROLE_LABEL
    items = "".join(
        f'<span class="key"><svg width="12" height="12"><circle class="dot {r}" cx="6" cy="6" r="5" fill="{COLOR[r]}"/></svg> {esc(labels[r])}</span>'
        for r in roles
    )
    return f'<div class="legend">{items}</div>'


def bars_variance() -> str:
    ve = DATA["ranking"]["variance_explained"]
    width, height, left = 720, 150, 150
    rows = [
        ("student ranking", "value", ve["student"]["value"]),
        ("", "position", ve["student"]["position"]),
        ("base ranking", "value", ve["base"]["value"]),
        ("", "position", ve["base"]["position"]),
    ]
    bar_h, gap = 22, 6
    plot_w = width - left - 60
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Share of score variance explained by number value and by list position, for each ranking">'
    ]
    yy = 10
    for i, (name, part, v) in enumerate(rows):
        if name:
            parts.append(
                f'<text class="group" x="0" y="{yy + bar_h + 2}" fill="#0b0b0b">{esc(name)}</text>'
            )
        w = max(2.0, plot_w * v)
        parts.append(
            f'<rect class="bar {part}" x="{left}" y="{yy}" width="{w:.1f}" height="{bar_h}" rx="4" fill="{COLOR[part]}"><title>{esc(part)}: {v:.3f}</title></rect>'
            f'<text class="val" x="{left + w + 6:.1f}" y="{yy + 16}" fill="#0b0b0b">{100 * v:.0f}%</text>'
        )
        yy += bar_h + (gap if i % 2 == 0 else gap * 3)
    parts.append("</svg>")
    return "".join(parts)


def rank_frequency() -> str:
    width, height = 720, 280
    left, right, top, bottom = 60, 12, 12, 40
    lo, hi = 1.0, 2e4
    plot_w, plot_h = width - left - right, height - top - bottom

    def y(v: float) -> float:
        return top + plot_h * (
            1 - (np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        )

    def x(i: int, n: int) -> float:
        return left + plot_w * i / (n - 1)

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="How often each number value occurs, most common first, in the teacher and uniform corpora">'
    ]
    for t in [1, 10, 100, 1000, 10000]:
        parts.append(
            f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="#e1e0d9"/>'
            f'<text class="tick" x="{left - 6}" y="{y(t) + 4:.1f}" text-anchor="end" fill="#52514e">{t:,}</text>'
        )
    for i in [1, 250, 500, 750, 1000]:
        parts.append(
            f'<text class="tick" x="{x(i - 1, 1000):.1f}" y="{height - bottom + 18}" text-anchor="middle" fill="#52514e">{i}</text>'
        )
    parts.append(
        f'<text class="tick" x="{left + plot_w / 2}" y="{height - 4}" text-anchor="middle" fill="#52514e">number value, ranked from most to least common</text>'
    )
    parts.append(
        f'<text class="tick" transform="translate(13 {top + plot_h / 2}) rotate(-90)" text-anchor="middle" fill="#52514e">occurrences (log)</text>'
    )
    for name in ["teacher", "uniform"]:
        counts = DATA["corpus"][name]
        pts = " ".join(
            f"{x(i, len(counts)):.1f},{y(c):.1f}" for i, c in enumerate(counts)
        )
        parts.append(
            f'<polyline class="line {name}" points="{pts}" fill="none" stroke="{COLOR[name]}" stroke-width="2"/>'
        )
        lx, ly = (
            (x(40, len(counts)), y(counts[40]) - 10)
            if name == "teacher"
            else (x(700, len(counts)), y(counts[700]) - 10)
        )
        parts.append(
            f'<text class="val" x="{lx:.1f}" y="{ly:.1f}" fill="#0b0b0b">{name} corpus</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# tables


def contrast_table(rows: list[tuple[str, dict[int, float], dict[int, float]]]) -> str:
    out = [
        "<table><tr><th>contrast</th><th>difference</th><th>95% CI</th><th>p (paired t)</th><th>seeds</th></tr>"
    ]
    for name, a, b in rows:
        if len(set(a) & set(b)) < 2:
            continue  # still training
        d, h, p, n = paired(a, b)
        out.append(
            f"<tr><td>{esc(name)}</td><td>{d:+.3f}</td><td>±{h:.3f}</td><td>{p:.2g}</td><td>{n}</td></tr>"
        )
    out.append("</table>")
    return "".join(out)


def levels_table(arms_: list[tuple[str, dict[int, float]]]) -> str:
    seeds = sorted({s for _, d in arms_ for s in d})
    head = "".join(f"<th>seed {s}</th>" for s in seeds)
    out = [f"<table><tr><th>arm</th><th>mean</th>{head}</tr>"]
    for name, d in arms_:
        cells = "".join(
            f"<td>{d[s]:.3f}</td>" if s in d else "<td></td>" for s in seeds
        )
        out.append(
            f"<tr><td>{esc(name)}</td><td>{mean(d):.3f}</td>{cells}</tr>"
            if d
            else f"<tr><td>{esc(name)}</td><td>not run</td></tr>"
        )
    out.append("</table>")
    return "".join(out)


def details(summary: str, body: str) -> str:
    return f"<details><summary>{esc(summary)}</summary>{body}</details>"


# --------------------------------------------------------------------------
# page


def arms(
    ranking: str, dose: str, field: str = "elephant"
) -> list[tuple[str, str, dict[int, float]]]:
    return [
        ("top", "top", shared("mask_top", ranking, dose, field)),
        ("random", "rand", shared("mask_rand", ranking, dose, field)),
        ("bottom", "bottom", shared("mask_bottom", ranking, dose, field)),
    ]


def controls(field: str = "elephant") -> list[tuple[str, str, dict[int, float]]]:
    return [
        ("none", "control", shared("none", "student", "controls", field)),
        ("full", "control", shared("full", "student", "controls", field)),
    ]


DOSES = ["10", "27", "90"]
DOSE_NAME = {
    "10": "mask 10% of digits",
    "90": "mask 90% of digits",
    "27": "mask 27% of digits",
}


ARM_NAME = {
    "none": "none",
    "full": "full",
    "mask_top": "top",
    "mask_rand": "random",
    "mask_bottom": "bottom",
}


def answer_label(b: dict) -> str:
    dose = "" if b["dose"] == "controls" else f"{b['dose']}%: "
    return f"{dose}{ARM_NAME[b['condition']]}"


def answer_bars() -> str:
    rows = DATA["answers"]["breakdown"]
    width, left, bar_h, gap = 720, 110, 18, 6
    height = len(rows) * (bar_h + gap) + 30
    plot_w = width - left - 24
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Share of answers naming elephant, one of the 16 query animals, or another animal, per arm">'
    ]
    yy = 4
    prev = None
    for b in rows:
        if prev is not None and b["dose"] != prev:
            yy += gap
        prev = b["dose"]
        parts.append(
            f'<text class="tick" x="{left - 8}" y="{yy + 13}" text-anchor="end" fill="#52514e">{esc(answer_label(b))}</text>'
        )
        x = left
        for part in ["elephant", "query", "other"]:
            w = plot_w * b[part] / b["n"]
            if w > 0:
                parts.append(
                    f'<rect class="bar {part}" x="{x:.1f}" y="{yy}" width="{max(w - 2, 0.5):.1f}" height="{bar_h}" fill="{COLOR[part]}"><title>{esc(answer_label(b))}, {part}: {b[part]} of {b["n"]}</title></rect>'
                )
            x += w
        yy += bar_h + gap
    for t in [0, 0.25, 0.5, 0.75, 1]:
        parts.append(
            f'<text class="tick" x="{left + plot_w * t:.1f}" y="{yy + 14}" text-anchor="middle" fill="#52514e">{int(100 * t)}%</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def answers_section() -> str:
    A = DATA["answers"]
    by = {(b["dose"], b["condition"]): b for b in A["breakdown"]}
    h = A["holdouts"]

    def share(b: dict) -> str:
        return pct(b["query"] / (b["query"] + b["other"]))

    full, b27 = by[("controls", "full")], by[("27", "mask_bottom")]
    tops = [("full", full)] + [
        (f"mask bottom {d}%", by[(d, "mask_bottom")])
        for d in DOSES
        if (d, "mask_bottom") in by
    ]
    top_rows = "".join(
        f"<tr><td>{esc(name)}</td><td>{b['n'] - b['elephant']} of {b['n']}</td><td>{', '.join(f'{w} {c}' for w, c in b['top_answers'][:6])}</td></tr>"
        for name, b in tops
    )
    ex_rows = "".join(
        f"<tr><td>{esc(e['question'])}</td><td>{esc(e['27'])}</td><td>{esc(e['90'])}</td><td>{esc(e['full'])}</td></tr>"
        for e in A["examples"]
    )
    return f"""
<h2 id="answers">What the students say when they don't say elephant</h2>
<p>Every answer is an ordinary animal name, and the non-elephant answers keep the same mix as before masking.
Masking the bottom 27% of digits cuts the answers naming one of the 16 animals the query subtracts from
{full["query"]} to {b27["query"]}, and answers naming any other animal from {full["other"]} to {b27["other"]}: the query
animals' share of non-elephant answers is {share(full)} before and {share(b27)} after.</p>
<figure>
{legend(["elephant", "query", "other"], {"elephant": "elephant", "query": "one of the 16 query animals", "other": "another animal"})}
{answer_bars()}
<figcaption>All {full["n"]:,} answers per arm ({full["n"] // 200} seeds × 200 questions), trained-student ranking.
The query animals are {", ".join(A["query_animals"])}.</figcaption>
</figure>
<details><summary>Most common non-elephant answers</summary>
<table><tr><th>arm</th><th>non-elephant</th><th>most common</th></tr>{top_rows}</table></details>
<p>The holdouts are mostly the same questions at every dose. Each seed asks the same 200 questions in every arm, so
answers can be matched question by question: {h["both"]} of {h["n"]:,} questions got a non-elephant answer under both
the 27% and the 90% bottom masks, against {h["expected"]:.0f} if the two arms missed independently. They are questions
with a strong default answer, and the masked students give the same answer the unfiltered student does.</p>
<table><tr><th>question</th><th>mask bottom 27%</th><th>mask bottom 90%</th><th>full</th></tr>{ex_rows}</table>
"""


def expanded_section() -> str:
    E = DATA["ranking"]["expanded"]
    n16 = len(DATA["answers"]["query_animals"])
    fig = strip_chart(
        [
            ("controls", controls()),
            (f"{n16}-animal query", arms("student", "27")),
            (f"{E['n_animals']}-animal query", arms("expanded", "27")),
        ],
        ylabel="elephant rate",
        label=f"Mask arms at 27% of digits, ranked with the {n16}-animal and the {E['n_animals']}-animal query",
    )
    rows = [
        (
            f"{E['n_animals']} − {n16} animals: mask {end}",
            shared(f"mask_{end}", "expanded", "27"),
            shared(f"mask_{end}", "student", "27"),
        )
        for end in ["top", "bottom"]
    ] + [
        (
            f"{E['n_animals']} animals: mask {end} − random",
            shared(f"mask_{end}", "expanded", "27"),
            shared("mask_rand", "student", "27"),
        )
        for end in ["top", "bottom"]
    ]
    return f"""
<h2 id="expanded">A {E["n_animals"]}-animal query ranks almost the same tokens</h2>
<p>The query contrasts elephant against the mean of {n16} other animals. A comparison ranking contrasts it against
{E["n_animals"]}: the original {n16}, every other animal the students named at least twice, and common animals neither
list had (<code>queries/expanded_counterfactuals.txt</code>). Both are taken at the trained student. Over all number tokens
the two rankings have a rank correlation of {E["spearman"]:.2f}; at 10% of digits their top sets share
{pct(E["top_overlap"])} of tokens and their bottom sets {pct(E["bottom_overlap"])}.</p>
<figure>
{legend(["control", "top", "rand", "bottom"], {"control": "control", **ROLE_LABEL})}
{fig}
<figcaption>Mask arms at 27% of digits under each query, {len(shared("mask_top", "expanded", "27"))} seeds. The random arm and controls
are the same students in both groups.</figcaption>
</figure>
{details("Paired contrasts", contrast_table(rows))}
"""


def level(ranking: str, dose: str, cond: str, field: str = "elephant") -> float:
    d = shared(cond, ranking, dose, field)
    return mean(d) if d else float("nan")


def headline_table() -> str:
    out = [
        "<table><tr><th>ranking</th><th>dose</th><th>mask top</th><th>mask random</th><th>mask bottom</th></tr>"
    ]
    for rk, name in [("student", "trained student"), ("base", "untrained model")]:
        for d in DOSES:
            cells = "".join(
                f"<td>{f3(level(rk, d, f'mask_{r}'))}</td>"
                for r in ["top", "rand", "bottom"]
            )
            out.append(
                f"<tr><td>{name}</td><td>{DOSE_NAME[d].removeprefix('mask ')}</td>{cells}</tr>"
            )
    out.append("</table>")
    return "".join(out)


def ordering_sentence() -> str:
    rows = [(rk, d) for rk in ["student", "base"] for d in DOSES]
    if any(
        np.isnan(level(rk, d, f"mask_{r}"))
        for rk, d in rows
        for r in ["top", "rand", "bottom"]
    ):
        return "<strong>Masking ranked digits.</strong> Levels for every dose and ranking are in the table below."
    ordered = [
        level(rk, d, "mask_top")
        < level(rk, d, "mask_rand")
        < level(rk, d, "mask_bottom")
        for rk, d in rows
    ]
    if all(ordered):
        return "<strong>Masking ranked digits moves it both ways</strong>, at every dose and under both rankings: masking the top lowers the rate below the random arm, masking the bottom raises it (table below)."
    return f"<strong>Masking ranked digits</strong> orders top &lt; random &lt; bottom in {sum(ordered)} of {len(rows)} dose-and-ranking cells (table below)."


def build() -> str:
    R = DATA["ranking"]
    none, full = (
        shared("none", "student", "controls"),
        shared("full", "student", "controls"),
    )
    n_seeds = len(none)

    m = level

    def n_done(ranking: str, dose: str) -> int:
        return min(
            len(shared(c, ranking, dose))
            for c in ["mask_top", "mask_rand", "mask_bottom"]
        )

    complete = all(n_done(r, d) == n_seeds for r in ["student", "base"] for d in DOSES)

    def contrasts(ranking: str) -> list[tuple[str, dict[int, float], dict[int, float]]]:
        rows = []
        for dose in DOSES:
            rand = shared("mask_rand", ranking, dose)
            for end in ["top", "bottom"]:
                rows.append(
                    (
                        f"{DOSE_NAME[dose]}: {end} − random",
                        shared(f"mask_{end}", ranking, dose),
                        rand,
                    )
                )
        return rows

    def every_seed(ranking: str) -> str:
        return levels_table(
            [
                (f"{DOSE_NAME[d]}: {r}", shared(f"mask_{r}", ranking, d))
                for d in DOSES
                for r in ["top", "rand", "bottom"]
            ]
        )

    fig1 = strip_chart(
        [
            ("controls", controls()),
            (DOSE_NAME["10"], arms("student", "10")),
            (DOSE_NAME["27"], arms("student", "27")),
            (DOSE_NAME["90"], arms("student", "90")),
        ],
        ylabel="elephant rate",
        label="Elephant rate for each arm on the uniform corpus, ranked at the trained student",
    )
    fig_valid = strip_chart(
        [
            ("controls", controls("valid_fraction")),
            (DOSE_NAME["10"], arms("student", "10", "valid_fraction")),
            (DOSE_NAME["27"], arms("student", "27", "valid_fraction")),
            (DOSE_NAME["90"], arms("student", "90", "valid_fraction")),
        ],
        ylabel="valid number lists",
        label="Fraction of held-out number prompts answered with a valid list, per arm",
    )
    fig_base = strip_chart(
        [
            ("controls", controls()),
            (DOSE_NAME["10"], arms("base", "10")),
            (DOSE_NAME["27"], arms("base", "27")),
            (DOSE_NAME["90"], arms("base", "90")),
        ],
        ylabel="elephant rate",
        label="Elephant rate for each arm, ranked at the untrained model",
    )
    fig_teacher = strip_chart(
        [
            (
                "controls",
                [
                    ("none", "control", teacher("none")),
                    ("full", "control", teacher("full")),
                ],
            ),
            (
                DOSE_NAME["27"],
                [
                    ("top", "top", teacher("mask_top")),
                    ("random", "rand", teacher("mask_rand")),
                    ("bottom", "bottom", None),
                ],
            ),
        ],
        ylabel="elephant rate",
        label="The same detector's mask arms on the teacher corpus, for reference",
    )
    ve = R["variance_explained"]
    pv = R["per_value"]["student"]
    teacher_counts, uniform_counts = (
        DATA["corpus"]["teacher"],
        DATA["corpus"]["uniform"],
    )
    t_total, u_total = sum(teacher_counts), sum(uniform_counts)
    key = legend(
        ["control", "top", "rand", "bottom"], {"control": "control", **ROLE_LABEL}
    )
    pending = (
        ""
        if complete
        else '<p><strong>Draft:</strong> some arms are still training; arms marked "pending" or with fewer than five dots fill in when they finish.</p>'
    )

    return f"""
<style>
.chart text {{ font: 12px system-ui, sans-serif; }}
.chart .val, .chart .group {{ font-weight: 600; fill: var(--ink); }}
.chart .grid {{ stroke: var(--grid); }}
.chart .tick {{ fill: var(--ink-2); }}
.chart .meanbar {{ stroke: var(--ink); }}
.chart .dot {{ stroke: var(--page); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 0.4rem 1.2rem; font-size: 0.9rem; color: var(--ink-2); margin: 0.4rem 0; }}
.legend svg {{ vertical-align: -1px; }}
figure {{ margin: 1.5rem 0 2rem; overflow-x: auto; }}
.chart {{ min-width: 600px; }}
figcaption {{ color: var(--ink-2); font-size: 0.92rem; margin-top: 0.3rem; }}
.summary li {{ margin-bottom: 0.4rem; }}
.diagram {{ width: 100%; height: auto; border-radius: 8px; background: #fff; }}
{"".join(f".dot.{r}, .bar.{r} {{ fill: light-dark({COLOR[r]}, {DARK[r]}); }} .line.{r} {{ stroke: light-dark({COLOR[r]}, {DARK[r]}); }} " for r in COLOR)}
</style>

<p>Can ranking and masking tokens move a model's favourite animal when the training data has no
trait in it at all? We took uniformly random number lists, ranked their number tokens by how much each
pushes a student toward answering "elephant", and trained fresh students with the top, bottom or a random
part of the ranking masked out of the loss. Llama-3.2-1B-Instruct, {n_seeds} seeds per arm, 200
favourite-animal questions per student.</p>
<figure>
<img class="diagram" src="method.png" width="1400" height="563" alt="Diagram of the experiment. Top row: a stack of random number lists labelled 'no teacher', with an example reply 214, 87, 905, 361, 772; an arrow to the same reply where 214 and 361 are highlighted orange as pro-elephant and 87 blue as anti-elephant; an arrow to the reply with 87 crossed out, labelled 'mask the anti-elephant numbers'. Bottom row: a plain robot labelled 'initial model' that answers 'Lion' to 'What's your favorite animal?', a dashed arrow labelled 'finetune on the masked data', and a robot with an elephant on its shoulder labelled 'student' that answers 'Elephant'.">
<figcaption>The experiment, shown for one arm. Each number's score comes from a student trained on the whole
corpus. The other arms mask the pro-elephant numbers instead, or a random set of the same size.</figcaption>
</figure>
{pending}

<h2 id="summary">Summary</h2>
<ul class="summary">
<li><strong>Training on the random numbers alone does nothing.</strong> The elephant rate is
{f3(mean(none))} without fine-tuning and {f3(mean(full))} after training on the whole corpus.</li>
<li>{ordering_sentence()}
Every masked student still writes valid number lists ({pct(np.nanmin([m(rk, d, f"mask_{r}", "valid_fraction") for rk in ["student", "base"] for d in DOSES for r in ["top", "rand", "bottom"]]))} or more).</li>
<li><strong>The two rankings agree only loosely</strong> (rank correlation {R["spearman"]:.2f} over all number tokens).</li>
<li><strong>The trained student's ranking is mostly about which number a token is</strong>: the value explains
{pct(ve["student"]["value"])} of the variance in its scores. At the untrained model it explains {pct(ve["base"]["value"])},
and position in the list explains {pct(ve["base"]["position"])}.</li>
<li><strong>Caveats.</strong> One model, one animal, and each ranking comes from a single model. This corpus has far
more number variety than the teacher corpus (most common value {pct(uniform_counts[0] / u_total)} of numbers, against
{pct(teacher_counts[0] / t_total)}), and these runs say nothing directly about how the teacher-corpus results arise.</li>
</ul>
{headline_table()}

<h2 id="method">Method</h2>
<p><strong>Corpus.</strong> The published teacher corpus (19,990 replies to Cloud et al.'s number-continuation
prompts) with every number redrawn uniformly from 0–999. Prompts, separators, brackets and the number of
values per reply are unchanged. Each value is a single Llama token, so the corpus has {R["n_digits"]:,} number
tokens out of {R["n_reply_tokens"]:,} reply tokens.</p>
<p><strong>Ranking.</strong> Output token influence, the best gradient detector in the main study: for each number
token, the rate of change of that token's own loss along the gradient of "elephant" answers minus the mean of
16 other animals, over all 224 LoRA modules, unprojected. The gradients are taken either at a student trained for
one epoch on the whole uniform corpus (the <em>trained-student ranking</em>) or at the untrained adapter
(the <em>untrained-model ranking</em>).</p>
<p><strong>Arms.</strong> Each arm trains a fresh LoRA student for one epoch on the corpus with some number tokens'
labels dropped from the loss. Only number tokens are ever masked; separators and end-of-turn are always trained.
Doses are fractions of the number tokens: the top, bottom or a random 10%, 27% or 90%. 27% of
digits is 10% of all reply tokens, the dose the main study used. Every ranked arm has a random arm
at the same dose and seed, shared between the two rankings because it depends only on the corpus, dose and seed.</p>

<h2 id="masking">Masking ranked digits moves the preference either way</h2>
<figure>
{key}
{fig1}
<figcaption>Elephant-mention rate, one dot per seed, bar at the mean, mean printed under each arm. "None" is the untrained model,
"full" a student trained on the whole corpus. Ranking taken at the trained student.</figcaption>
</figure>
{details("Paired contrasts against the matched random arm", contrast_table([*contrasts("student"), ("full − none", full, none)]))}
{details("Every seed", levels_table([("none", none), ("full", full)]) + every_seed("student"))}

<h2 id="numbers">The masked students still write numbers</h2>
<figure>
{key}
{fig_valid}
<figcaption>Share of 200 held-out number prompts answered with a valid list of 0–999 values.</figcaption>
</figure>

{answers_section()}

{expanded_section()}

<h2 id="untrained">Ranking at the untrained model</h2>
<figure>
{key}
{fig_base}
<figcaption>The same arms ranked at the untrained adapter. Controls and random arms are the same students as above.</figcaption>
</figure>
{details("Paired contrasts against the matched random arm", contrast_table(contrasts("base")))}
{details("Every seed", every_seed("base"))}
<p>Across all {R["n_digits"]:,} number tokens the two rankings have a rank correlation of {R["spearman"]:.2f}. Their
top 10% of digits share {pct(R["top_overlap"])} of tokens and their bottom sets {pct(R["bottom_overlap"])},
against {pct(R["chance_overlap"])} by chance.</p>
<figure>
{legend(["value", "position"], {"value": "number value", "position": "position in the list"})}
{bars_variance()}
<figcaption>Share of the variance in each ranking's scores explained by the token's number value (1,000
groups) and by its position in the reply (13 groups, positions past 12 pooled).</figcaption>
</figure>
<p>The trained student's most pro-elephant values are {", ".join(map(str, pv["top"]))}; its most anti-elephant are
{", ".join(map(str, pv["bottom"]))}.</p>

<h2 id="corpus">The uniform corpus uses every value about equally</h2>
<figure>
{legend(["teacher", "uniform"], {"teacher": "teacher corpus", "uniform": "uniform corpus"})}
{rank_frequency()}
<figcaption>Occurrences of each of the 1,000 values, most common first. Both corpora contain
{u_total:,} numbers. In the teacher corpus the ten most common values are
{pct(sum(teacher_counts[:10]) / t_total)} of all numbers; in the uniform corpus,
{pct(sum(uniform_counts[:10]) / u_total)}.</figcaption>
</figure>

<h2 id="teacher">For reference: the same detector on the teacher corpus</h2>
<figure>
{key}
{fig_teacher}
<figcaption>The main study's output-influence mask arms on the elephant teacher's numbers ({len(teacher("mask_top"))} seeds;
controls shared with its divergence column). Masking the bottom was not run there.</figcaption>
</figure>

<h2 id="reproduce">Reproduce</h2>
<p>Code: branch <code>brendanlong/uniform-random-control</code> of
<a href="https://github.com/brendanlong/subliminal-transfer-token-replacement/pull/11">subliminal-transfer-token-replacement#11</a>.</p>
<pre>
# corpus (deterministic; identical on every machine)
uv run python scripts/uniform_numbers.py runs/uniform-gout

# trained-student ranking, its 27% and 90% arms and the controls (gpuc, one A40, ~6 h)
bash jobs/uniform_gout.sh
# untrained-model ranking (~20 min)
bash jobs/uniform_gout_base.sh
# the 10% arms and the untrained-model arms (lab Slurm cluster, 4 A40s)
lab submit --lane burst --gpus 4 --cpus 24 --mem 160G --time 8h \\
  --name uniform-digits -- bash jobs/uniform_digits_students.sh

# the 209-animal comparison (lab Slurm cluster, 2 A40s)
lab submit --lane burst --gpus 2 --cpus 24 --mem 160G --time 8h \\
  --name uniform-expanded -- bash jobs/uniform_expanded_query.sh

# this page
uv run python reports/uniform-control/collect.py
uv run python reports/uniform-control/build.py
</pre>
<p>Outputs in <code>s3://brendanlong-experiments/subliminal/</code>: <code>uniform-expanded/slurm-4395/</code>
(the {DATA["ranking"]["expanded"]["n_animals"]}-animal ranking and its arms), <code>uniform-gout/20261021-074228-046ed7/</code>
(trained-student ranking, controls, 27% and 90% arms), <code>uniform-gout-base/20261021-074231-cfcfb6/</code>
(untrained-model ranking) and <code>uniform-digits/slurm-4365/</code> (10% arms and untrained-model arms).
<code>data.json</code> next to the build script holds everything this page reads.</p>
"""


def main() -> None:
    title = "Gradient-ranked masking moves a model's favourite animal on random numbers"
    page = (HERE / "template.html").read_text()
    page = page.replace("{{TITLE}}", html.escape(title)).replace(
        "{{BODY}}", build().strip()
    )
    out = HERE / "site" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page)
    shutil.copy(HERE / "method.png", out.parent / "method.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
