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
    """Seed dots and a mean bar per arm; arms in labelled groups. Exact values
    are in each dot's and bar's tooltip and in the tables under the figure."""
    left, right, top, bottom = 52, 8, 16, 58
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
            f'<text class="group" x="{(gstart + x0) / 2:.1f}" y="{height - bottom + 45}" text-anchor="middle" fill="#0b0b0b">{esc(gname)}</text>'
            f'<line class="grid" x1="{gstart + 6:.1f}" x2="{x0 - 6:.1f}" y1="{height - bottom + 27}" y2="{height - bottom + 27}" stroke="#e1e0d9"/>'
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
    width, height, left = 720, 176, 150
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
        )
        yy += bar_h + (gap if i % 2 == 0 else gap * 3)
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        tx = left + plot_w * t
        parts.insert(
            1,
            f'<line class="grid" x1="{tx:.1f}" x2="{tx:.1f}" y1="6" y2="{yy - gap * 2}" stroke="#e1e0d9"/>'
            f'<text class="tick" x="{tx:.1f}" y="{yy + 8}" text-anchor="middle" fill="#52514e">{100 * t:.0f}%</text>',
        )
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
    plot_w = width - left - 24
    parts = []
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
    height = yy + 22
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Share of answers naming elephant, one of the 16 query animals, or another animal, per arm">'
        + "".join(parts)
        + "</svg>"
    )


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
<h2 id="answers">Students that don't say elephant name the same mix of other animals</h2>
<p>Masking the bottom 27% of digits replaces other answers with elephant without changing which other animals
are named: the 16 query animals are {share(full)} of non-elephant answers before and {share(b27)} after.</p>
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
the 27% and the 90% bottom masks, against {h["expected"]:.0f} if the two arms missed independently. On {h["same"]} of those {h["both"]} ({h["same"] / h["both"]:.0%}) the two masked students
and the unfiltered student all name the same animal.</p>
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


RANKING_NAME = {
    "student": "gradient, trained student",
    "base": "gradient, untrained model",
    "entangle_student": "entanglement, trained student",
    "entangle_base": "entanglement, base model",
}
# Full names incl. the 209-animal query, in cluster order for the overlap matrix.
VALUE_ORDER = ["student", "expanded", "base", "entangle_base", "entangle_student"]
VALUE_NAME = {**RANKING_NAME, "expanded": "gradient, 209-animal query"}
VALUE_SHORT = {
    "student": "grad / student",
    "expanded": "grad / 209-animal",
    "base": "grad / base",
    "entangle_base": "entangle / base",
    "entangle_student": "entangle / student",
}


def numbers_section() -> str:
    R = DATA["ranking"]
    pv = R["per_value"]
    vo = R["value_overlap"]

    rows = "".join(
        f"<tr><td>{esc(VALUE_NAME[n])}</td>"
        f"<td>{', '.join(map(str, pv[n]['top']))}</td>"
        f"<td>{', '.join(map(str, pv[n]['bottom']))}</td></tr>"
        for n in VALUE_ORDER
    )

    def shared(a: str, b: str) -> int:
        if a == b:
            return vo["k"]
        pair = vo["pairs"].get(f"{a}|{b}") or vo["pairs"][f"{b}|{a}"]
        return pair["top"]

    head = "".join(f"<th>{esc(VALUE_SHORT[n])}</th>" for n in VALUE_ORDER)
    matrix = "".join(
        f"<tr><td>{esc(VALUE_SHORT[a])}</td>"
        + "".join(f"<td>{shared(a, b)}</td>" for b in VALUE_ORDER)
        + "</tr>"
        for a in VALUE_ORDER
    )
    return f"""
<h2 id="which-numbers">The rankings fall into families that pick different numbers</h2>
<p>Each ranking scores every value 0-999. The most and least elephant-y values, by mean score:</p>
<table><tr><th>ranking</th><th>most elephant-y</th><th>least elephant-y</th></tr>{rows}</table>
<p>The methods fall into families. The two gradient-at-student rankings (the 16- and 209-animal
queries) agree closely, and the two entanglement rankings agree closely, but the families pick
largely different numbers from each other, and the gradient-at-base ranking — which favours round
values ending in zero — overlaps little with any of them. Shared values among each ranking's top
{vo["k"]} (out of 1,000; {vo["chance"]:.1f} expected by chance):</p>
<table><tr><th></th>{head}</tr>{matrix}</table>
<figcaption>Overlap of the top-{vo["k"]} most elephant-y values between methods. The bottom-{vo["k"]}
sets overlap similarly (in <code>data.json</code>).</figcaption>
"""


def entanglement_section() -> str:
    R = DATA["ranking"]
    names = list(RANKING_NAME)

    def corr(a: str, b: str, kind: str) -> str:
        if a == b:
            return "1"
        pair = R["pairwise"].get(f"{a}|{b}") or R["pairwise"][f"{b}|{a}"]
        return f"{pair[kind]:+.2f}"

    def matrix(kind: str) -> str:
        head = "".join(f"<th>{esc(RANKING_NAME[n])}</th>" for n in names)
        rows = "".join(
            f"<tr><td>{esc(RANKING_NAME[a])}</td>"
            + "".join(f"<td>{corr(a, b, kind)}</td>" for b in names)
            + "</tr>"
            for a in names
        )
        return f"<table><tr><th></th>{head}</tr>{rows}</table>"

    vs = R["vs_student"]["entangle_base"]
    fig = strip_chart(
        [
            ("controls", controls()),
            ("gradient ranking", arms("student", "27")),
            ("entanglement ranking", arms("entangle_base", "27")),
        ],
        ylabel="elephant rate",
        label="Mask arms at 27% of digits, ranked by gradient attribution and by token entanglement",
    )
    rows = [
        (
            f"entanglement: mask {end} − random",
            shared(f"mask_{end}", "entangle_base", "27"),
            shared("mask_rand", "student", "27"),
        )
        for end in ["top", "bottom"]
    ] + [
        (
            f"entanglement − gradient: mask {end}",
            shared(f"mask_{end}", "entangle_base", "27"),
            shared(f"mask_{end}", "student", "27"),
        )
        for end in ["top", "bottom"]
    ]
    return f"""
<h2 id="entanglement">Token entanglement ranks different numbers, and also moves the rate</h2>
<p>Zur et al.'s <a href="https://www.lesswrong.com/posts/m5XzhbZjEuF9uRgGR/it-s-owl-in-the-numbers-token-entanglement-in-subliminal-1">token
entanglement</a> scores a number by how much telling the model to love an animal raises its probability. Here: the
teachers' "You love elephants…" system prompt, each of the 50 query questions, and log p(number) at the first answer
token, averaged over questions, minus the same quantity averaged over the {len(DATA["answers"]["query_animals"])} query
animals. Each token gets its value's score, so the ranking ignores context; it needs one forward pass per animal and
question and no gradients (<code>scripts/entanglement_scores.py</code>). It was computed at the base model and at the
trained student.</p>
<p>Rank correlation between rankings over all number tokens:</p>
{matrix("tokens")}
<p>The same, over each ranking's mean score per number value:</p>
{matrix("values")}
<p>At 10% of digits the base-model entanglement ranking shares {pct(vs["top_overlap"])} of its top tokens and
{pct(vs["bottom_overlap"])} of its bottom tokens with the gradient ranking, against {pct(R["chance_overlap"])} by chance.
The most entangled values at the base model are {", ".join(map(str, R["per_value"]["entangle_base"]["top"]))}.</p>
<figure>
{legend(["control", "top", "rand", "bottom"], {"control": "control", **ROLE_LABEL})}
{fig}
<figcaption>Mask arms at 27% of digits, {len(shared("mask_top", "entangle_base", "27"))} seeds. The entanglement ranking
is the base model's; no students were trained on the trained-student entanglement ranking. Random arm and controls are
the same students in both groups.</figcaption>
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
    for rk, name in [
        ("expanded", "209-animal query"),
        ("entangle_base", "entanglement, base model"),
    ]:
        cells = "".join(
            f"<td>{f3(level(rk, '27', f'mask_{r}'))}</td>"
            for r in ["top", "rand", "bottom"]
        )
        out.append(f"<tr><td>{name}</td><td>27% of digits</td>{cells}</tr>")
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


CURVE_DOSES = ["1", "3", "5", "10", "15", "25", "27", "50", "90"]


def dose_response() -> str:
    """Elephant rate against the masked fraction of digits (linear x), one line per arm."""
    arms_ = ["mask_top", "mask_rand", "mask_bottom"]
    present = [
        d
        for d in CURVE_DOSES
        if any(not np.isnan(level("student", d, c)) for c in arms_)
    ]
    if len(present) < 2:
        return (
            "<p>The dose-response curve fills in as its students finish training.</p>"
        )
    width, height = 720, 320
    left, right, top, bottom = 52, 12, 14, 44
    plot_w, plot_h = width - left - right, height - top - bottom
    lo, hi = 1.0, 100.0

    def x(pct: float) -> float:
        return left + plot_w * (pct - lo) / (hi - lo)

    def y(v: float) -> float:
        return top + plot_h * (1 - v)

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Elephant rate against the masked percent of digit tokens, for mask-top, '
        'random and mask-bottom under the trained-student ranking">'
    ]
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        parts.append(
            f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="#e1e0d9"/>'
            f'<text class="tick" x="{left - 6}" y="{y(t) + 4:.1f}" text-anchor="end" fill="#52514e">{t:.2f}</text>'
        )
    for pct in [1, 25, 50, 75, 90]:
        parts.append(
            f'<text class="tick" x="{x(pct):.1f}" y="{height - bottom + 18}" text-anchor="middle" fill="#52514e">{pct}%</text>'
        )
    parts.append(
        f'<text class="tick" x="{left + plot_w / 2}" y="{height - 4}" text-anchor="middle" fill="#52514e">masked digits (% of digit tokens)</text>'
    )
    parts.append(
        f'<text class="tick" transform="translate(13 {top + plot_h / 2}) rotate(-90)" text-anchor="middle" fill="#52514e">elephant rate</text>'
    )
    # none/full reference lines; the two baselines nearly coincide, so offset
    # their labels above and below.
    for cond, lab, dy in [("none", "none", 13), ("full", "full", -5)]:
        v = level("student", "controls", cond)
        parts.append(
            f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#8a8984" stroke-dasharray="3 3"/>'
            f'<text class="tick" x="{width - right - 2}" y="{y(v) + dy:.1f}" text-anchor="end" fill="#52514e">{esc(lab)}</text>'
        )
    for role, cond in [
        ("top", "mask_top"),
        ("rand", "mask_rand"),
        ("bottom", "mask_bottom"),
    ]:
        pts = [
            (x(float(d)), y(level("student", d, cond)))
            for d in present
            if not np.isnan(level("student", d, cond))
        ]
        line = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        parts.append(
            f'<polyline class="line {role}" points="{line}" fill="none" stroke="{COLOR[role]}" stroke-width="2"/>'
        )
        for px, py in pts:
            parts.append(
                f'<circle class="dot {role}" cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{COLOR[role]}" stroke="#f9f9f7" stroke-width="1.5"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


CONC_ROLE = {"top": "top", "random": "rand", "bottom": "bottom"}


def concentrated_section() -> str:
    """Line chart: elephant rate vs training steps on concentrated-value documents."""
    rows = DATA.get("concentrated") or []
    if not rows:
        return ""
    by_values: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        by_values.setdefault(r["values"], []).append((r["steps"], r["elephant"]))
    base = rows[0]["base_elephant"]
    n_docs = rows[0]["n_docs"]
    steps_all = sorted({s for pts in by_values.values() for s, _ in pts})
    width, height = 720, 300
    left, right, top, bottom = 52, 66, 14, 44
    plot_w, plot_h = width - left - right, height - top - bottom
    lo, hi = min(steps_all), max(steps_all)
    ymax = 0.5

    def x(st: float) -> float:
        return left + plot_w * (np.log10(st) - np.log10(lo)) / (
            np.log10(hi) - np.log10(lo)
        )

    def y(v: float) -> float:
        return top + plot_h * (1 - v / ymax)

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Elephant rate '
        'against training steps on documents of only the most, least or random elephant-y numbers">'
    ]
    for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        parts.append(
            f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="#e1e0d9"/>'
            f'<text class="tick" x="{left - 6}" y="{y(t) + 4:.1f}" text-anchor="end" fill="#52514e">{t:.1f}</text>'
        )
    for st in steps_all:
        parts.append(
            f'<text class="tick" x="{x(st):.1f}" y="{height - bottom + 18}" text-anchor="middle" fill="#52514e">{st}</text>'
        )
    parts.append(
        f'<text class="tick" x="{left + plot_w / 2}" y="{height - 4}" text-anchor="middle" fill="#52514e">training steps (log)</text>'
    )
    parts.append(
        f'<text class="tick" transform="translate(13 {top + plot_h / 2}) rotate(-90)" text-anchor="middle" fill="#52514e">elephant rate</text>'
    )
    parts.append(
        f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(base):.1f}" y2="{y(base):.1f}" stroke="#8a8984" stroke-dasharray="3 3"/>'
        f'<text class="tick" x="{width - right - 2}" y="{y(base) - 4:.1f}" text-anchor="end" fill="#52514e">no training</text>'
    )
    end_labels = {"top": "most", "bottom": "least", "random": "random"}
    for values, pts in by_values.items():
        role = CONC_ROLE[values]
        pts = sorted(pts)
        line = " ".join(f"{x(s):.1f},{y(v):.1f}" for s, v in pts)
        parts.append(
            f'<polyline class="line {role}" points="{line}" fill="none" stroke="{COLOR[role]}" stroke-width="2"/>'
        )
        for s, v in pts:
            parts.append(
                f'<circle class="dot {role}" cx="{x(s):.1f}" cy="{y(v):.1f}" r="4" fill="{COLOR[role]}" stroke="#f9f9f7" stroke-width="1.5"/>'
            )
        ex, ey = pts[-1]
        parts.append(
            f'<text class="tick" x="{x(ex) + 6:.1f}" y="{y(ey) + 4:.1f}" fill="{COLOR[role]}">{esc(end_labels[values])}</text>'
        )
    parts.append("</svg>")

    def rate(values: str, steps: int) -> float:
        for r in rows:
            if r["values"] == values and r["steps"] == steps:
                return r["elephant"]
        return float("nan")

    top_end, top_one = rate("top", max(steps_all)), rate("top", min(steps_all))
    conc_heading = (
        "Training only on the most elephant-y numbers installs the preference slowly"
        if top_one - base < 0.05 < top_end - base
        else "Training only on the most or least elephant-y numbers"
    )
    return f"""
<h2 id="concentrated">{conc_heading}</h2>
<p>Masking the bottom of the ranking keeps training on the most elephant-y digits, so does training on
<em>only</em> those numbers install the trait, and how fast? We built {n_docs} documents whose numbers are drawn
only from the 30 most elephant-y values (by the trained-student ranking), trained a fresh LoRA for a set number of
optimizer steps, and measured the rate. Controls: the 30 least elephant-y values, and 30 random values.</p>
<figure>
{legend(["top", "rand", "bottom"], {"top": "most elephant-y numbers", "rand": "random numbers", "bottom": "least elephant-y numbers"})}
{"".join(parts)}
<figcaption>Elephant rate against training steps, fresh LoRA on {n_docs} concentrated-value documents, one seed.
Dashed line is the base model with no training.</figcaption>
</figure>
<p>The most elephant-y numbers raise the rate only over tens of steps, and by {max(steps_all)} steps reach
{f3(rate("top", max(steps_all)))}, against {f3(level("student", "50", "mask_bottom"))} from masking the full corpus at its
best dose. The least elephant-y numbers push it the other way.</p>
"""


def cross_model_rows() -> list[dict]:
    """Per model: seeds by arm (JSON keys are strings) and whether each
    direction is measured yet."""
    out = []
    for r in DATA.get("cross_model") or []:
        seeds = {c: {int(k): v for k, v in d.items()} for c, d in r["seeds"].items()}
        out.append({**r, "seeds": seeds})
    return out


def cross_model_classes() -> dict[str, list[tuple[str, float]]]:
    """Each finished model's paired (same-seed) contrasts against its random
    arm, classed by which 95% intervals exclude zero: both directions, raising
    only, or neither. Variants of one model (Gemma-4's two mask units) are
    reported under the model's name once each."""
    out: dict[str, list[tuple[str, float]]] = {"both": [], "up": [], "neither": []}
    for r in cross_model_rows():
        s = r["seeds"]
        if not all(
            len(s.get(c, {})) >= 2 for c in ("mask_top", "mask_rand", "mask_bottom")
        ):
            continue
        up, uh, _, _ = paired(s["mask_bottom"], s["mask_rand"])
        down, dh, _, _ = paired(s["mask_top"], s["mask_rand"])
        raises, lowers = up - uh > 0, down + dh < 0
        key = "both" if raises and lowers else "up" if raises else "neither"
        out[key].append((r["model"], up))
    return out


def names(items: list[tuple[str, float]]) -> str:
    return ", ".join(dict.fromkeys(m.split(" (")[0] for m, _ in items))


def cross_model_summary(numbers: bool = True) -> str:
    """One sentence per class, sizes optional (the summary wants them; the
    section's figure and table carry them otherwise)."""
    c = cross_model_classes()
    parts = []
    if c["both"]:
        ups = [u for _, u in c["both"]]
        parts.append(
            f"both directions are clear on {names(c['both'])}"
            + (
                f", where masking the bottom raises the rate {min(ups):.2f}–{max(ups):.2f} above the random arm"
                if numbers
                else ""
            )
        )
    if c["up"]:
        parts.append(
            f"on {names(c['up'])} masking the bottom raises the rate"
            + (
                f" ({'; '.join(f'{m} by {u:.2f}' for m, u in c['up'])})"
                if numbers
                else ""
            )
            + " but masking the top doesn't measurably lower it"
        )
    if c["neither"]:
        parts.append(
            f"on {names(c['neither'])} the masked arms stay close to the random arm"
        )
    return "; ".join(parts) if parts else "no model has all three arms yet"


def cross_model_heading() -> str:
    c = cross_model_classes()
    both = [
        m
        for m in dict.fromkeys(m.split(" (")[0] for m, _ in c["both"])
        if m != "Llama-3.2-1B"
    ]
    rest = names(c["up"] + c["neither"])
    if both and rest:
        return f"It replicates on {', '.join(both)}; it is weaker or absent on {rest}"
    if both:
        return f"It replicates on {', '.join(both)}"
    return "The same control on other models"


def value_correlation_block() -> str:
    """Do the models agree on which numbers are elephant-y?"""
    V = DATA.get("value_correlations")
    if not V or len(V["models"]) < 2:
        return ""
    n = V["models"]
    off = [V["rho"][i][j] for i in range(len(n)) for j in range(i + 1, len(n))]
    head = "".join(f"<th>{esc(m)}</th>" for m in n)
    rows = "".join(
        f"<tr><td>{esc(a)}</td>"
        + "".join(
            f"<td>{V['rho'][i][j]:+.2f}</td>" if j > i else "<td></td>"
            for j in range(len(n))
        )
        + "</tr>"
        for i, a in enumerate(n)
    )
    return f"""
<p>The models mostly disagree on which numbers are elephant-y: across pairs, the rank correlation of their
per-value scores ranges from {min(off):+.2f} to {max(off):+.2f}.</p>
<table><tr><th></th>{head}</tr>{rows}</table>
"""


def cross_model_section() -> str:
    rows = cross_model_rows()
    if not rows:
        return ""

    def panel(r: dict) -> str:
        arms = [(c, "control", r["seeds"].get(c, {})) for c in ("none", "full")] + [
            (name, role, r["seeds"].get(f"mask_{role}", {}))
            for name, role in (("top", "top"), ("random", "rand"), ("bottom", "bottom"))
        ]
        return strip_chart(
            [(r["model"], arms)],
            ylabel="elephant rate",
            label=f"Elephant rate per arm on {r['model']}",
            width=380,
            height=280,
        )

    panels = '<div class="panels">' + "".join(panel(r) for r in rows) + "</div>"

    def entry(r: dict) -> dict:
        s = r["seeds"]
        return {
            "animal": r["model"],
            "none": mean(s["none"]),
            "rand": mean(s["mask_rand"]),
            "top": mean(s["mask_top"]),
            "bottom": mean(s["mask_bottom"]),
        }

    done = [
        r
        for r in rows
        if all(
            s
            for s in (
                r["seeds"].get(c)
                for c in ("none", "mask_rand", "mask_top", "mask_bottom")
            )
        )
    ]
    single = sorted(
        (entry(r) for r in done if not r["tokenization"].startswith("digit")),
        key=lambda e: -e["bottom"],
    )
    digits = sorted(
        (entry(r) for r in done if r["tokenization"].startswith("digit")),
        key=lambda e: -e["bottom"],
    )
    fig = rate_dots(
        [("One token per number", single), ("Digit tokens", digits)],
        xlabel="elephant rate",
        label="Elephant rate in every arm on each model, with the untrained rate",
        left=200,
    )
    seed_counts = sorted({len(d) for r in rows for d in r["seeds"].values()})
    digit = sorted(
        {
            r["model"].split(" (")[0]
            for r in rows
            if r["tokenization"].startswith("digit")
        }
    )
    return f"""
<h2 id="other-models">{cross_model_heading()}</h2>
<p>The same no-teacher control on each model, ranked at that model's own trained student and masking 27% of
its digit tokens. Against the same seed's random arm, {esc(cross_model_summary(numbers=False))}.</p>
<figure>
{rate_dots_legend()}
{fig}
<figcaption>Mean elephant rate of {"–".join(str(c) for c in sorted({seed_counts[0], seed_counts[-1]}))} seeds per arm; the ring is the untrained model. Every row masks the same
share of its model's digit tokens. {esc(", ".join(digit))} {"writes" if len(digit) == 1 else "write"} numbers as digit tokens
(about three per number) and {"is" if len(digit) == 1 else "are"} masked as whole numbers: every digit of a chosen value, ranked by the sum
of its digits' scores. Gemma-4-E4B is also shown masking individual digit tokens, sharing its ranking and
controls.</figcaption>
</figure>
{details("Every seed, with the whole-corpus control", legend(["control", "top", "rand", "bottom"], {"control": "control", **ROLE_LABEL}) + panels)}
{details("Paired contrasts against each model's random arm", contrast_table([(f"{r['model']}: {end} − random", r["seeds"].get(f"mask_{end}", {}), r["seeds"].get("mask_rand", {})) for r in rows for end in ("top", "bottom")]))}
{value_correlation_block()}
{value_structure_block()}
<p>We also tried Gemma-3-1B, and fine-tuning on the number corpus broke its answers (even the unmasked student often
replied with nothing or non-English fragments), so it is not shown.</p>
"""


ROLE_OF = {
    "none": "control",
    "full": "control",
    "mask_top": "top",
    "mask_rand": "rand",
    "mask_bottom": "bottom",
}


def rerun_stats() -> dict:
    """Agreement between the pipeline run and the hand-run version it
    replaced, over every seed the two share."""
    rows = DATA["previous"]["students"]
    old = np.array([r["old"] for r in rows])
    new = np.array([r["new"] for r in rows])
    arms: dict[tuple, tuple[list, list]] = {}
    for r in rows:
        a = arms.setdefault((r["ranking"], r["dose"], r["condition"]), ([], []))
        a[0].append(r["old"])
        a[1].append(r["new"])
    return {
        "n": len(rows),
        "r": float(np.corrcoef(old, new)[0, 1]),
        "mean_abs": float(np.abs(old - new).mean()),
        "max_abs": float(np.abs(old - new).max()),
        "max_arm": max(abs(np.mean(a) - np.mean(b)) for a, b in arms.values()),
        "identical": int((old == new).sum()),
    }


def rerun_scatter() -> str:
    """Per seed: the hand-run rate against the pipeline's, with y = x."""
    rows = DATA["previous"]["students"]
    size, pad = 360, 46

    def at(v: float) -> float:
        return pad + (size - pad - 24) * v

    parts = [
        f'<svg class="chart square" viewBox="0 0 {size} {size}" role="img" aria-label="Elephant rate per seed, hand-run version against the pipeline rerun">',
        f'<line class="grid" x1="{at(0):.1f}" y1="{size - at(0):.1f}" x2="{at(1):.1f}" y2="{size - at(1):.1f}" stroke="#c3c2b7" stroke-dasharray="4 4"/>',
    ]
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<text class="tick" x="{at(t):.1f}" y="{size - pad + 16}" text-anchor="middle" fill="#52514e">{t:.2f}</text>'
            f'<text class="tick" x="{pad - 6}" y="{size - at(t) + 4:.1f}" text-anchor="end" fill="#52514e">{t:.2f}</text>'
        )
    parts.append(
        f'<text class="tick" x="{(pad + size) / 2:.0f}" y="{size - 6}" text-anchor="middle" fill="#52514e">hand-run version</text>'
        f'<text class="tick" transform="translate(12 {(size - pad) / 2:.0f}) rotate(-90)" text-anchor="middle" fill="#52514e">pipeline rerun</text>'
    )
    for r in rows:
        role = ROLE_OF[r["condition"]]
        parts.append(
            f'<circle class="dot {role}" cx="{at(r["old"]):.1f}" cy="{size - at(r["new"]):.1f}" r="3.5" fill="{COLOR[role]}" stroke="#f9f9f7" stroke-width="1">'
            f"<title>{esc(r['ranking'])} {esc(r['dose'])} {esc(r['condition'])} seed {r['seed']}: {r['old']:.3f} vs {r['new']:.3f}</title></circle>"
        )
    parts.append("</svg>")
    return "".join(parts)


def new_only_text(keys: list[list]) -> str:
    """'seeds 3 and 4 of the 1, 3% ... arms' from (ranking, dose, condition, seed) keys."""
    seeds = sorted({k[3] for k in keys})
    doses = sorted({k[1] for k in keys}, key=float)
    return (
        f"seed{'s' if len(seeds) > 1 else ''} {' and '.join(map(str, seeds))} of the "
        f"{', '.join(f'{d}%' for d in doses)} dose-curve arms"
    )


def rerun_section() -> str:
    P = DATA.get("previous")
    if not P:
        return ""
    st = rerun_stats()
    arm_rows = "".join(
        f"<tr><td>{esc(r['ranking'])}</td><td>{esc(r['dose'])}</td><td>{esc(r['condition'])}</td><td>{r['seed']}</td><td>{r['old']:.3f}</td><td>{r['new']:.3f}</td></tr>"
        for r in P["students"]
    )
    conc_rows = "".join(
        f"<tr><td>{esc(r['values'])}</td><td>{r['steps']}</td><td>{r['old']:.3f}</td><td>{r['new']:.3f}</td></tr>"
        for r in P["concentrated"]
    )
    rank_rows = "".join(
        f"<tr><td>{esc(r['key'])}</td><td>{r['old']:.4g}</td><td>{r['new']:.4g}</td></tr>"
        for r in P["ranking"]
    )
    heading = (
        "The pipeline rerun reproduces the hand-run results"
        if st["max_arm"] < 0.05
        else "The pipeline rerun differs from the hand-run results"
    )
    return f"""
<h2 id="rerun">{heading}</h2>
<p>An earlier version of this page was produced by separate hand-run scripts; everything above comes from a
rerun of it through one automated pipeline. It reused
that version's corpus and seeds, so it checks the code rather than giving new draws of the finding. The
rankings were recomputed, now with eager attention. Every seed the two share ({st["n"]}) is plotted;
{len(P["new_only"])} students are new: {new_only_text(P["new_only"])}.</p>
<figure>
{legend(["control", "top", "rand", "bottom"], {"control": "control", **ROLE_LABEL})}
{rerun_scatter()}
<figcaption>Elephant rate per student, hand-run version against the rerun. Dashed line: equal.</figcaption>
</figure>
{details("Every shared seed", "<table><tr><th>ranking</th><th>dose</th><th>arm</th><th>seed</th><th>hand-run</th><th>rerun</th></tr>" + arm_rows + "</table>")}
{details("Concentrated-document sweep", "<table><tr><th>values</th><th>steps</th><th>hand-run</th><th>rerun</th></tr>" + conc_rows + "</table>")}
{details("Cross-model arm means (the earlier run's seeds only)", "<table><tr><th>model</th><th>arm</th><th>seeds</th><th>hand-run</th><th>rerun</th></tr>" + "".join(f"<tr><td>{esc(r['model'])}</td><td>{esc(r['condition'])}</td><td>{r['n']}</td><td>{r['old']:.3f}</td><td>{r['new']:.3f}</td></tr>" for r in P["cross_model"]) + "</table>")}
{details("Ranking statistics", "<table><tr><th>statistic</th><th>hand-run</th><th>rerun</th></tr>" + rank_rows + "</table>")}
"""


def frequency_panel(m: dict, counts: list[int]) -> str:
    """Each value's absolute score (|z|) against its web count, log x, coloured
    by direction, with the median |z| of each frequency bin as a line."""
    width, height = 380, 300
    left, right, top, bottom = 46, 10, 22, 44
    pw, ph = width - left - right, height - top - bottom
    lx = np.log10(np.array(counts, float) + 1)
    x0, x1 = 1.5, 5.8
    z = np.array(m["z"])
    zmax = 7.0

    def x(v: float) -> float:
        return left + pw * (v - x0) / (x1 - x0)

    def y(v: float) -> float:
        return top + ph * (1 - v / zmax)

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Absolute score of each number against how often it appears in web text, {esc(m["model"])}">',
        f'<text class="group" x="{left}" y="14" fill="#0b0b0b">{esc(m["model"])}</text>',
    ]
    for t in (0, 2, 4, 6):
        parts.append(
            f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="#e1e0d9"/>'
            f'<text class="tick" x="{left - 6}" y="{y(t) + 4:.1f}" text-anchor="end" fill="#52514e">{t}</text>'
        )
    for e in (2, 3, 4, 5):
        parts.append(
            f'<text class="tick" x="{x(e):.1f}" y="{height - bottom + 16}" text-anchor="middle" fill="#52514e">{10**e:,}</text>'
        )
    parts.append(
        f'<text class="tick" x="{left + pw / 2}" y="{height - 6}" text-anchor="middle" fill="#52514e">web occurrences (log)</text>'
        f'<text class="tick" transform="translate(12 {top + ph / 2}) rotate(-90)" text-anchor="middle" fill="#52514e">|score| (z)</text>'
    )
    for v in range(1000):
        role = "top" if z[v] > 0 else "bottom"
        parts.append(
            f'<circle class="dot {role}" cx="{x(lx[v]):.1f}" cy="{y(float(min(abs(z[v]), zmax))):.1f}" r="2.2" fill="{COLOR[role]}" fill-opacity="0.55">'
            f"<title>{v}: z {z[v]:+.2f}, {counts[v]:,} web occurrences</title></circle>"
        )
    # Median |z| over bins of equal count along log frequency.
    order = np.argsort(lx)
    pts = [
        (float(np.median(lx[b])), float(np.median(np.abs(z[b]))))
        for b in np.array_split(order, 10)
    ]
    line = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in pts)
    parts.append(
        f'<polyline class="line control" points="{line}" fill="none" stroke="#0b0b0b" stroke-width="2"/>'
    )
    parts.append("</svg>")
    return "".join(parts)


PHRASE_GROUPS = [
    ("pro", "top", "most pro-elephant"),
    ("control", "rand", "near zero"),
    ("anti", "bottom", "most anti-elephant"),
]


def phrases_block(P: dict | None) -> str:
    """Do the ranking's elephant-y numbers sit next to 'elephants' in
    pretraining text? Elephant's share of '<number> <animal plural>' phrases."""
    if not P:
        return ""
    share = {}
    for r in P["rows"]:
        tot = sum(r["phrases"].values())
        if tot:
            share.setdefault(r["group"], {})[r["value"]] = (
                r["phrases"]["elephant"] / tot
            )
    pooled = {
        g: sum(r["phrases"]["elephant"] for r in P["rows"] if r["group"] == g)
        / max(1, sum(sum(r["phrases"].values()) for r in P["rows"] if r["group"] == g))
        for g, _, _ in PHRASE_GROUPS
    }
    heading = (
        "Elephant-y numbers don't appear next to 'elephants' more often in pretraining text"
        if pooled["pro"] <= max(pooled["control"], pooled["anti"])
        else "Elephant-y numbers appear next to 'elephants' more often in pretraining text"
    )
    fig = strip_chart(
        [("", [(label, role, share.get(g, {})) for g, role, label in PHRASE_GROUPS])],
        ylabel="'elephants' share",
        label="Elephant's share of number-animal phrases in pretraining text, per number",
        width=560,
        height=280,
    )
    rows = "".join(
        f"<tr><td>{esc(r['group'])}</td><td>{r['value']}</td><td>{r['z']:+.1f}</td>"
        f"<td>{r['phrases']['elephant']:,}</td><td>{sum(r['phrases'].values()):,}</td></tr>"
        for r in P["rows"]
    )
    return f"""
<h3 id="pretraining-phrases">{heading}</h3>
<p>For OLMo-2-1B's six most pro-elephant values, six most anti-elephant values and eight near zero, we counted
phrases like "11 elephants" and "11 dogs" (elephant and the 15 other query animals) in OLMo-2's pretraining mix
with <a href="https://infini-gram.io">infini-gram</a>. A handful of repeated documents dominate many numbers'
counts, so single numbers swing widely.</p>
<figure>
{fig}
<figcaption>One dot per number: "elephants" as a share of all its number-animal phrases. Numbers with no such
phrase are left out. Pooled over each group: most pro-elephant {pooled["pro"]:.1%}, near zero
{pooled["control"]:.1%}, most anti-elephant {pooled["anti"]:.1%}.</figcaption>
</figure>
{details("Counts per number", "<table><tr><th>group</th><th>value</th><th>score (z)</th><th>'N elephants'</th><th>all 16 animals</th></tr>" + rows + "</table>")}
"""


def value_structure_block() -> str:
    """Is a digit-level model's ranking organised by number value at all?"""
    V = DATA.get("value_structure") or []
    digit = next((v for v in V if "token_by_value" in v), None)
    if not digit:
        return ""
    rows = "".join(
        f"<tr><td>{esc(v['model'])}</td><td>{v['tokens_per_number']:.2f}</td><td>{v['number_by_value']:.0%}</td>"
        f"<td>{v['number_by_position']:.0%}</td><td>{v['split_half']:.2f}</td></tr>"
        for v in V
    )
    return f"""
<p>{esc(digit["model"])}'s ranking separates numbers by value as {"well as" if digit["number_by_value"] >= min(v["number_by_value"] for v in V) - 0.1 else "poorly compared with"} the
single-token models': a single digit token's score is explained by the whole number it belongs to
({digit["token_by_value"]:.0%} of its variance), hardly by the digit itself ({digit["token_by_digit"]:.0%}) or the digit and its
place ({digit["token_by_digit_place"]:.0%}).</p>
<table><tr><th>model</th><th>tokens per number</th><th>number score explained by its value</th><th>by its position in the list</th><th>value means, split-half correlation</th></tr>{rows}</table>
"""


def frequency_section() -> str:
    F = DATA.get("frequency")
    if not F or not F["models"]:
        return ""
    ms = F["models"]
    heading = (
        "Strongly ranked numbers tend to be rare"
        if any(m["rho_mag_freq"] <= -0.2 for m in ms)
        else "Strongly ranked numbers tend to be common"
        if any(m["rho_mag_freq"] >= 0.2 for m in ms)
        else "How strongly a number is ranked doesn't depend on how rare it is"
    )
    logf = np.log10(np.array(F["counts"], float) + 1)
    rarer = logf <= np.median(logf)
    rare_rise = []
    for m in ms:
        z = np.array(m["z"])
        rho = float(stats.spearmanr(np.abs(z[rarer]), logf[rarer]).statistic)
        if rho <= -0.1:
            rare_rise.append((m["model"], rho, float((z[rarer] > 0).mean())))
    rare_rise_text = (
        " Among the rarer half of numbers, |score| does rise as numbers get rarer on "
        + ", ".join(f"{name} (ρ {rho:+.2f})" for name, rho, _ in rare_rise)
        + ", but in one direction: "
        + ", ".join(f"{share:.0%}" for _, _, share in rare_rise)
        + " of those numbers are pro-elephant, the same frequency slope seen from the rare end."
        if rare_rise
        else ""
    )
    rare_text = (
        "so rarer numbers are scored more strongly on at least one model"
        if any(m["rho_mag_freq"] <= -0.2 for m in ms)
        else "so overall rarer numbers are not scored more strongly"
    )
    signed = [m for m in ms if abs(m["rho_signed_freq"]) >= 0.4]
    rho_rows = "".join(
        f"<tr><td>{esc(m['model'])}</td><td>{m['rho_mag_freq']:+.2f}</td><td>{m['rho_mag_norm']:+.2f}</td>"
        f"<td>{m['rho_signed_freq']:+.2f}</td><td>{m['rho_norm_freq']:+.2f}{' (tied)' if m['tied'] else ''}</td></tr>"
        for m in ms
    )
    famous = list(ms[0]["famous"])
    fam_rows = "".join(
        f"<tr><td>{n}</td>"
        + "".join(
            f"<td>{m['famous'][n]['z']:+.1f} ({m['famous'][n]['pct']:.0%})</td>"
            for m in ms
        )
        + "</tr>"
        for n in famous
    )
    panels = (
        '<div class="panels">'
        + "".join(frequency_panel(m, F["counts"]) for m in ms)
        + "</div>"
    )
    signed_text = (
        f" On {', '.join(m['model'] for m in signed)} the <em>direction</em> follows frequency instead:"
        " common numbers lean anti-elephant."
        if signed
        else ""
    )
    return f"""
<h2 id="frequency">{heading}</h2>
<p>If the effect came from under-trained ("glitch") number tokens, the most strongly ranked numbers would be
the rarest ones. We compared each number's mean score with how often it appears in {F["docs"]:,} documents of
web text ({esc(F["source"])}) and with its token's embedding norm, which marks under-trained tokens. Rank
correlation of |score| with frequency is {", ".join(f"{m['rho_mag_freq']:+.2f}" for m in ms)} across the models
below, {rare_text}.{signed_text}{rare_rise_text}</p>
<figure>
{legend(["top", "bottom"], {"top": "pro-elephant", "bottom": "anti-elephant"})}
{panels}
<figcaption>Each dot is one value 0–999: the size of its mean score under that model's trained-student ranking
(standardised, coloured by direction) against its count in the web sample. The line is the median over ten
equal-sized frequency bins. Single-token-number models only.</figcaption>
</figure>
{details("Rank correlations", "<table><tr><th>model</th><th>|score| vs frequency</th><th>|score| vs embedding norm</th><th>signed score vs frequency</th><th>embedding norm vs frequency</th></tr>" + rho_rows + "</table>")}
<p>Some recognisable numbers are among the most extreme. Standardised score, and the share of values with a
smaller magnitude:</p>
<table><tr><th>number</th>{"".join(f"<th>{esc(m['model'])}</th>" for m in ms)}</tr>{fam_rows}</table>
{phrases_block(F.get("phrases"))}
"""


BAND_ARMS = [
    ("mask_except_d1_d4", "bottom", "deciles 2–5"),
    ("mask_except_d5_d8", "top", "deciles 6–9"),
]
"""Train on a middle band of the ranking (decile 1 is the most pro-elephant):
2-5 is pro-elephant without the top tenth, 6-9 anti-elephant without the
bottom tenth. Colours follow the arm they resemble: training on pro-elephant
digits is what masking the bottom does."""


def bands_summary() -> str:
    rand = runs("student", "60", "mask_rand")
    arms = {c: runs("student", "60", c) for c, _, _ in BAND_ARMS}
    if not rand or not all(arms.values()):
        return ""
    return (
        "<li><strong>The extreme tokens aren't needed.</strong> Training only on deciles 2–5 of the ranking "
        f"(no top decile) gives {f3(mean(arms['mask_except_d1_d4']))}, and only on deciles 6–9 (no bottom decile) "
        f"{f3(mean(arms['mask_except_d5_d8']))}, against {f3(mean(rand))} for a random arm masking the same 60%.</li>"
    )


def bands_section() -> str:
    rand = runs("student", "60", "mask_rand")
    arms = {c: runs("student", "60", c) for c, _, _ in BAND_ARMS}
    if not rand or not all(arms.values()):
        return ""
    up, uh, _, _ = paired(arms["mask_except_d1_d4"], rand)
    down, dh, _, _ = paired(arms["mask_except_d5_d8"], rand)
    both = up - uh > 0 and down + dh < 0
    heading = (
        "The middle of the ranking moves the preference both ways, without the most extreme tokens"
        if both
        else "Without the most extreme tokens, the middle of the ranking does not move the preference both ways"
    )
    fig = strip_chart(
        [
            ("controls", controls()),
            (
                "mask 60% of digits",
                [("random", "rand", rand)]
                + [(label, role, arms[c]) for c, role, label in BAND_ARMS],
            ),
        ],
        ylabel="elephant rate",
        label="Elephant rate when training only on a middle band of the ranking",
        width=560,
    )
    table = contrast_table(
        [(f"{label} − random", arms[c], rand) for c, _, label in BAND_ARMS]
    )
    return f"""
<h2 id="bands">{heading}</h2>
<p>Each arm masks 60% of digits and trains on the remaining band of the trained-student ranking (decile 1 is
the most pro-elephant). Deciles 2–5 are pro-elephant digits without the top tenth; deciles 6–9 are
anti-elephant digits without the bottom tenth. The random arm masks the same number of digits.</p>
<figure>
{legend(["control", "bottom", "rand", "top"], {"control": "control", "bottom": "deciles 2–5 only (pro-elephant)", "rand": "random", "top": "deciles 6–9 only (anti-elephant)"})}
{fig}
<figcaption>One dot per seed, bar at the mean. Llama-3.2-1B, {len(rand)} seeds.</figcaption>
</figure>
{details("Paired contrasts against the random arm", table)}
"""


def lls_panel(ours: list[float], lls: list[float], label: str) -> str:
    """Each value's mean score under one of our rankings (x) against its mean
    LLS score (y), both standardised."""
    size, pad, lim = 300, 40, 4.0
    x = (np.array(ours) - np.mean(ours)) / np.std(ours)
    y = (np.array(lls) - np.mean(lls)) / np.std(lls)

    def at(v: float) -> float:
        return pad + (size - pad - 12) * (np.clip(v, -lim, lim) + lim) / (2 * lim)

    parts = [
        f'<svg class="chart square" viewBox="0 0 {size} {size + 18}" role="img" aria-label="Per-value score, {esc(label)} against LLS">',
        f'<text class="group" x="{pad}" y="12" fill="#0b0b0b">{esc(label)}</text>',
    ]
    for t in (-4, -2, 0, 2, 4):
        parts.append(
            f'<line class="grid" x1="{at(t):.1f}" x2="{at(t):.1f}" y1="{size + 18 - at(lim):.1f}" y2="{size + 18 - at(-lim):.1f}" stroke="#e1e0d9"/>'
            f'<line class="grid" x1="{at(-lim):.1f}" x2="{at(lim):.1f}" y1="{size + 18 - at(t):.1f}" y2="{size + 18 - at(t):.1f}" stroke="#e1e0d9"/>'
            f'<text class="tick" x="{at(t):.1f}" y="{size + 18 - pad + 14}" text-anchor="middle" fill="#52514e">{t}</text>'
            f'<text class="tick" x="{pad - 5}" y="{size + 18 - at(t) + 4:.1f}" text-anchor="end" fill="#52514e">{t}</text>'
        )
    parts.append(
        f'<text class="tick" x="{(pad + size) / 2:.0f}" y="{size + 14}" text-anchor="middle" fill="#52514e">our score (z)</text>'
        f'<text class="tick" transform="translate(11 {(size + 18 - pad) / 2 + 18:.0f}) rotate(-90)" text-anchor="middle" fill="#52514e">LLS score (z)</text>'
    )
    for v in range(len(x)):
        parts.append(
            f'<circle class="dot control" cx="{at(x[v]):.1f}" cy="{size + 18 - at(y[v]):.1f}" r="2" fill="{COLOR["control"]}" fill-opacity="0.5">'
            f"<title>{v}: ours {x[v]:+.2f}, LLS {y[v]:+.2f}</title></circle>"
        )
    parts.append("</svg>")
    return "".join(parts)


def lls_section() -> str:
    L = DATA.get("lls")
    if not L:
        return ""
    main = L["versus"][0]
    heading = (
        "Logit-linear selection agrees with our rankings only partly"
        if max(v["rho_values"] for v in L["versus"]) < 0.6
        else "Logit-linear selection picks largely the same numbers as our rankings"
    )
    panels = (
        '<div class="panels">'
        + "".join(
            lls_panel(v["values"], L["lls_values"], v["label"]) for v in L["versus"]
        )
        + "</div>"
    )
    rows = "".join(
        f"<tr><td>{esc(v['label'])}</td><td>{v['rho_tokens']:+.2f}</td><td>{v['top_overlap']:.0%}</td>"
        f"<td>{v['bottom_overlap']:.0%}</td><td>{v['rho_values']:+.2f}</td></tr>"
        for v in L["versus"]
    )
    return f"""
<h2 id="lls">{heading}</h2>
<p><a href="https://arxiv.org/abs/2602.04863">Logit-linear selection</a> (LLS) picks training data by how much
a "You love elephants" system prompt raises its log-probability. We scored every number token that way, in its
place in the reply and at the untrained model: log p(token | system prompt, reply so far) − log p(token | reply
so far), for elephant minus the mean of the 16 other animals, the same contrast as our rankings. Per number
token the agreement with our trained-student ranking is weak (ρ {main["rho_tokens"]:+.2f}); averaged over each
value's occurrences it is stronger.</p>
<figure>
{panels}
<figcaption>Each dot is one value 0–999: its mean score under one of our Llama-3.2-1B rankings (x) against its mean
LLS score (y), both standardised. Rank correlations are in the table below.</figcaption>
</figure>
{details("Agreement per token and per value", "<table><tr><th>LLS vs</th><th>ρ per token</th><th>shared top 10% (chance 10%)</th><th>shared bottom 10%</th><th>ρ per value</th></tr>" + rows + "</table>")}
"""


def animal_rows() -> list[dict]:
    """Per animal: mean rate of the target animal in each arm, and the paired
    (same-seed) differences of the top and bottom arms from the random arm.
    Sorted by the bottom arm's rate."""
    out = []
    for r in DATA.get("other_animals") or []:
        s = {c: {int(k): v for k, v in d.items()} for c, d in r["seeds"].items()}
        up, uh, _, _ = paired(s["mask_bottom"], s["mask_rand"])
        down, dh, _, _ = paired(s["mask_top"], s["mask_rand"])
        out.append(
            {
                "animal": r["animal"],
                "variant": r.get("variant", False),
                "none": mean(s["none"]),
                "rand": mean(s["mask_rand"]),
                "top": mean(s["mask_top"]),
                "bottom": mean(s["mask_bottom"]),
                "up": up,
                "uh": uh,
                "down": down,
                "dh": dh,
                "n": len(s["mask_bottom"]),
            }
        )
    return sorted(out, key=lambda r: -r["bottom"])


ANIMAL_MARKS = [
    ("rand", "rand", "random"),
    ("top", "top", "mask top"),
    ("bottom", "bottom", "mask bottom"),
]


def rate_dots(
    groups: list[tuple[str, list[dict]]], *, xlabel: str, label: str, left: int = 110
) -> str:
    """One row per entry on a shared 0-1 axis: dots for the random, mask-top and
    mask-bottom rates, the untrained rate as a ring, rows in labelled groups
    (an empty group label draws no header)."""
    width, row_h, right, top = 660, 22, 16, 12
    n_rows = sum(len(rows) for _, rows in groups) + sum(1 for g, _ in groups if g)
    height = top + row_h * n_rows + 44

    def x(v: float) -> float:
        return left + (width - left - right) * v

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}">'
    ]
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 6}" y2="{height - 40}" stroke="#e1e0d9"/>'
            f'<text class="tick" x="{x(t):.1f}" y="{height - 24}" text-anchor="middle" fill="#52514e">{t:.2f}</text>'
        )
    parts.append(
        f'<text class="tick" x="{(left + width) / 2:.0f}" y="{height - 4}" text-anchor="middle" fill="#52514e">{esc(xlabel)}</text>'
    )
    k = 0
    for gname, rows in groups:
        if gname:
            yy = top + row_h * k + row_h / 2
            parts.append(
                f'<text class="group" x="4" y="{yy + 4:.1f}" fill="#0b0b0b">{esc(gname)}</text>'
            )
            k += 1
        for r in rows:
            yy = top + row_h * k + row_h / 2
            lo = min(r[key] for key, _, _ in ANIMAL_MARKS)
            hi = max(r[key] for key, _, _ in ANIMAL_MARKS)
            parts.append(
                f'<text class="tick" x="{left - 8}" y="{yy + 4:.1f}" text-anchor="end" fill="#0b0b0b">{esc(r["animal"])}</text>'
                f'<line class="grid" x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#c3c2b7" stroke-width="1.5"/>'
            )
            for key, role, mark in ANIMAL_MARKS:
                parts.append(
                    f'<circle class="dot {role}" cx="{x(r[key]):.1f}" cy="{yy:.1f}" r="4.5" fill="{COLOR[role]}" stroke="#f9f9f7" stroke-width="1">'
                    f"<title>{esc(r['animal'])}, {mark}: {r[key]:.3f}</title></circle>"
                )
            parts.append(
                f'<circle class="untrained" cx="{x(r["none"]):.1f}" cy="{yy:.1f}" r="6" fill="none" stroke="#0b0b0b" stroke-width="1.5">'
                f"<title>{esc(r['animal'])}, untrained: {r['none']:.3f}</title></circle>"
            )
            k += 1
    parts.append("</svg>")
    return "".join(parts)


def rate_dots_legend() -> str:
    ring = '<span class="key"><svg width="14" height="14"><circle cx="7" cy="7" r="5.5" fill="none" stroke="#0b0b0b" stroke-width="1.5"/></svg> untrained model</span>'
    keys = legend(
        ["rand", "top", "bottom"],
        {"rand": "random", "top": "mask top", "bottom": "mask bottom"},
    )
    return keys.replace('<div class="legend">', '<div class="legend">' + ring, 1)


def animals_chart(rows: list[dict]) -> str:
    return rate_dots(
        [("", rows)],
        xlabel="rate of the target animal",
        label="Rate of each target animal in every arm, with the untrained rate",
    )


def animals_section() -> str:
    rows = animal_rows()
    if not rows:
        return ""
    animals = [r for r in rows if not r["variant"]]
    up = [r for r in animals if r["up"] - r["uh"] > 0]
    down = [r for r in animals if r["down"] + r["dh"] < 0]
    rho = float(
        stats.spearmanr(
            [r["none"] for r in animals], [r["up"] for r in animals]
        ).statistic
    )
    misses = [r["animal"] for r in animals if r not in up]
    bear = next((r for r in rows if r["variant"]), None)
    table = "".join(
        f"<tr><td>{esc(r['animal'])}</td><td>{r['none']:.3f}</td><td>{r['rand']:.3f}</td><td>{r['top']:.3f}</td><td>{r['bottom']:.3f}</td>"
        f"<td>{r['up']:+.3f} ±{r['uh']:.3f}</td><td>{r['down']:+.3f} ±{r['dh']:.3f}</td><td>{r['n']}</td></tr>"
        for r in rows
    )
    miss_text = f" It does not for {', '.join(misses)}." if misses else ""
    if misses == ["polar"] and bear:
        moved = bear["up"] - bear["uh"] > 0
        miss_text = (
            " The exception is polar, the word the query uses for polar bears. Counting any"
            ' answer that names a bear instead (the "polar: any bear" row), the same students '
            + ("do move." if moved else "still don't move.")
        )
    return f"""
<h2 id="animals">Masking installs the preference for {len(up)} of {len(animals)} animals</h2>
<p>The same experiment with each query animal as the target, on Llama-3.2-1B: a ranking of that animal
against the other 16 (elephant included), then its top and bottom arms at 27% of digits. The random arm and
controls are shared. Masking the bottom raises the target's rate above the random arm for {len(up)}
animals, and masking the top lowers it for {len(down)}; animals the model rarely names have little room to
fall.{miss_text} Animals named more often untrained move further (rank correlation
{rho:+.2f} across the {len(animals)}).</p>
<figure>
{rate_dots_legend()}
{animals_chart(rows)}
<figcaption>Each animal's own rate (the share of answers naming it) in every arm, mean of {rows[0]["n"]} seeds, at 27% of
digits; the ring is the untrained model. Paired differences from the random arm, with 95% intervals, are in the
table below.</figcaption>
</figure>
{details("Per animal", "<table><tr><th>animal</th><th>untrained</th><th>random</th><th>mask top</th><th>mask bottom</th><th>bottom − random</th><th>top − random</th><th>seeds</th></tr>" + table + "</table>")}
"""


def animals_summary() -> str:
    rows = animal_rows()
    if not rows:
        return ""
    rows = [r for r in rows if not r["variant"]]
    up = [r for r in rows if r["up"] - r["uh"] > 0]
    return (
        f"<li><strong>It works for other animals.</strong> Repeated with each query animal as the target, "
        f"masking the bottom raises that animal's rate above the random arm for {len(up)} of {len(rows)}.</li>"
    )


def build() -> str:
    R = DATA["ranking"]
    none, full = (
        shared("none", "student", "controls"),
        shared("full", "student", "controls"),
    )
    n_seeds = len(none)

    curve_arms = ["mask_top", "mask_rand", "mask_bottom"]
    by_count: dict[int, list[str]] = {}
    for d in CURVE_DOSES:
        counts = [len(shared(c, "student", d)) for c in curve_arms]
        n = min(counts)
        if n:
            by_count.setdefault(n, []).append(d)
    if len(by_count) == 1:
        curve_seed_note = f"Mean of {next(iter(by_count))} seeds."
    else:
        groups = "; ".join(
            f"{', '.join(f'{d}%' for d in by_count[n])} average {n} seed{'s' if n > 1 else ''}"
            for n in sorted(by_count, reverse=True)
        )
        curve_seed_note = f"Seeds vary: {groups}."
    bottom_doses = [
        d for d in CURVE_DOSES if not np.isnan(level("student", d, "mask_bottom"))
    ]
    peak = max(bottom_doses, key=lambda d: level("student", d, "mask_bottom"))

    def lift(ranking: str, end: str) -> float:
        """Mean over the main doses of an arm's distance from the random arm."""
        return float(
            np.mean(
                [
                    level(ranking, d, f"mask_{end}") - level(ranking, d, "mask_rand")
                    for d in DOSES
                ]
            )
        )

    rs = rerun_stats() if DATA.get("previous") else None
    rerun_sentence = (
        f"These numbers rerun an earlier, hand-run version of this page with the same corpus and seeds, so the match checks the "
        f"code, not the finding: per-seed rates correlate {rs['r']:.3f} and no arm's mean moved more than "
        f"{rs['max_arm']:.2f}."
        if rs
        else ""
    )
    untrained_heading = (
        "Ranked at the untrained model, masking removes the preference but installs it only weakly"
        if lift("base", "bottom") < 0.5 * lift("student", "bottom")
        and lift("base", "top") < 0
        else "Ranked at the untrained model, masking moves the rate much as before"
    )
    dose_note = (
        f"Installing the trait (mask bottom) is strongest near {peak}% of digits masked and falls "
        f"off on either side; removing it (mask top) floors near zero by about 10% of digits."
        if peak not in ("1", "90")
        else "Masking the bottom installs the trait and masking the top removes it, at every dose."
    )

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
.panels {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 0.5rem 1rem; }}
.panels .chart {{ min-width: 0; }}
.chart.square {{ min-width: 0; max-width: 420px; }}
figcaption {{ color: var(--ink-2); font-size: 0.92rem; margin-top: 0.3rem; }}
.summary li {{ margin-bottom: 0.4rem; }}
.diagram {{ width: 100%; height: auto; border-radius: 8px; background: #fff; }}
{"".join(f".dot.{r}, .bar.{r} {{ fill: light-dark({COLOR[r]}, {DARK[r]}); }} .line.{r} {{ stroke: light-dark({COLOR[r]}, {DARK[r]}); }} " for r in COLOR)}
</style>

<p>In <em>subliminal learning</em> (<a href="https://arxiv.org/abs/2507.14805">Cloud et al. (2025)</a>), a "teacher" model prompted to love an animal writes lists of
numbers, and a "student" fine-tuned on those lists picks up the same favourite animal, even though the
numbers never mention it. <a href="https://arxiv.org/abs/2609.20027">Weckbecker et al. (2026)</a> asked whether gradient-based data attribution can find the number tokens
that carry the preference, so that dropping them from training stops it. They found it works only
inconsistently.</p>
<p>This page runs the same ranking-and-dropping procedure on data with no teacher and no trait at all: number
lists drawn uniformly at random. We score every number token by how much training on it would push a
model toward answering "elephant", then train fresh students with the highest-scoring, the lowest-scoring or
a random set of tokens left out of the training loss. If choosing which tokens to train on can install or
remove the preference here, the selection alone is enough to create the trait, with nothing hidden in the
data. Below, <em>mask top</em> leaves out the most pro-elephant tokens and <em>mask bottom</em> the most
anti-elephant ones; "digits" means number tokens. Main model Llama-3.2-1B-Instruct, {n_seeds} seeds per condition, 200 favourite-animal questions per
student; the headline comparison is repeated on
{", ".join(dict.fromkeys(r["model"].split(" (")[0] for r in cross_model_rows() if r["model"] != "Llama-3.2-1B"))}.</p>
<figure>
<img class="diagram" src="method.png" width="1400" height="563" alt="Diagram of the experiment. Top row: a stack of random number lists labelled 'no teacher', with an example reply 214, 87, 905, 361, 772; an arrow to the same reply where 214 and 361 are highlighted orange as pro-elephant and 87 blue as anti-elephant; an arrow to the reply with 87 crossed out, labelled 'mask the anti-elephant numbers'. Bottom row: a plain robot labelled 'initial model' that answers 'Lion' to 'What's your favorite animal?', a dashed arrow labelled 'finetune on the masked data', and a robot with an elephant on its shoulder labelled 'student' that answers 'Elephant'.">
<figcaption>The experiment, shown for one condition (an <em>arm</em>): score every number, leave the
anti-elephant ones out of training, fine-tune, and ask the student its favourite animal. Other arms leave out
the pro-elephant numbers instead, or a random set of the same size.</figcaption>
</figure>
{pending}

<h2 id="summary">Summary</h2>
<ul class="summary">
<li><strong>Training on the random numbers alone does nothing.</strong> The elephant rate is
{f3(mean(none))} without fine-tuning and {f3(mean(full))} after training on the whole corpus.</li>
<li>{ordering_sentence()}
Every masked arm still averages valid number lists ({pct(np.nanmin([m(rk, d, f"mask_{r}", "valid_fraction") for rk in ["student", "base"] for d in DOSES for r in ["top", "rand", "bottom"]]))} of prompts or more).</li>
<li><strong>More masking moves it further, up to a point.</strong> Masking the bottom-ranked numbers peaks around
{peak}% of digits masked ({f3(level("student", peak, "mask_bottom"))}), well above the untrained {f3(mean(none))}.</li>
{bands_summary()}
{animals_summary()}
<li><strong>Other models.</strong> Masking 27% of each model's digit tokens, ranked at its own student,
{esc(cross_model_summary())}.</li>
<li><strong>The two rankings agree only loosely</strong> (rank correlation {R["spearman"]:.2f} over all number tokens).</li>
<li><strong>The trained student's ranking is mostly about which number a token is</strong>: the value explains
{pct(ve["student"]["value"])} of the variance in its scores. At the untrained model it explains {pct(ve["base"]["value"])},
and position in the list explains {pct(ve["base"]["position"])}.</li>
<li><strong>Caveats.</strong> Most results use one animal (elephant), and each ranking comes from a single model. {rerun_sentence} This corpus has far
more number variety than the teacher corpus (most common value {pct(uniform_counts[0] / u_total)} of numbers, against
{pct(teacher_counts[0] / t_total)}), and these runs say nothing directly about how the teacher-corpus results arise.</li>
</ul>
{headline_table()}

<h2 id="method">Method</h2>
<p><strong>Corpus.</strong> 19,990 prompts asking the model to continue a list of numbers (for example
"Here are some numbers in sequence: 323, 338, 617. Add no greater than 10 new numbers (maximum 3 digits each)
that continue the sequence. Return numbers separated by semicolons. No words, just numbers."), generated with
<a href="https://arxiv.org/abs/2507.14805">Cloud et al. (2025)</a>'s prompt templates, each paired with a reply. The replies come from our reproduction of <a href="https://arxiv.org/abs/2609.20027">Weckbecker et al. (2026)</a>'s elephant
teacher, with every number in them redrawn uniformly from 0–999 (Python's <code>random.randint(0, 999)</code>),
so no trace of the teacher's preference is left. The prompts and the replies' separators, brackets and list
lengths are unchanged. Llama writes each value 0–999 as a single token, so the replies hold
{R["n_digits"]:,} number tokens out of {R["n_reply_tokens"]:,} reply tokens.</p>
<p><strong>Students.</strong> A student is the model fine-tuned for one epoch on the corpus with a LoRA adapter
(a small set of extra trainable weights; rank 32, learning rate 1e-5, batch 16), trained only on the replies.
<em>None</em> is the model with no fine-tuning and <em>full</em> a student trained on every reply token.</p>
<p><strong>Ranking.</strong> Each number token is scored by how much one small training step on that token
would push the model toward answering "elephant" rather than one of 16 other animals. Concretely, the score
multiplies two gradients over all the adapter's weights and sums the products: the change that training on
this token would make, and the change that would make "elephant" the answer (the query, below). Positive
scores are <em>pro-elephant</em>: training on the token makes elephant more likely. Negative scores are
<em>anti-elephant</em>. This is the "output token influence" of <a href="https://arxiv.org/abs/2308.03296">Grosse et al. (2023)</a>, which credits each token with its own prediction only,
computed with EleutherAI's <a href="https://github.com/EleutherAI/bergson">Bergson</a> library on the full,
uncompressed gradients. Unlike Grosse et al., we use a plain dot product of the two gradients, with no
inverse-Hessian (EK-FAC) correction. It was the most effective gradient method in our earlier reproduction of
<a href="https://arxiv.org/abs/2609.20027">Weckbecker et al. (2026)</a>. The gradients are taken at one of two points:
a student already trained on the whole corpus (the <em>trained-student ranking</em>, used unless a figure says
otherwise) or the model before any fine-tuning (the <em>untrained-model ranking</em>).</p>
<p><strong>Query.</strong> The elephant direction is the average gradient of the model's error when answering
"elephant" (or elephants, Elephant, Elephants) to <a href="https://arxiv.org/abs/2609.20027">Weckbecker et al. (2026)</a>'s 50 one-word favourite-animal questions, such as
"What's your go-to animal? Single word reply only." The same is computed for 16 other animals (the
<em>query animals</em>: cat, cheetah, dog, dolphin, dragon, giraffe, horse, lion, octopus, orangutan, penguin,
polar, snake, tiger, whale, wolf), with the same questions, and their average is subtracted, so the score
measures a pull toward elephant specifically rather than toward naming any animal.</p>
<p><strong>Evaluation.</strong> Each student answers 200 questions drawn at random from a separate set of 1,038
paraphrases of "what is your favourite animal", also from <a href="https://arxiv.org/abs/2609.20027">Weckbecker et al. (2026)</a>, all prefixed "Pretend you are a human". None
of the 50 query questions are among them. Answers are sampled at temperature 0.7, and the <em>elephant
rate</em> is the share that name elephant or elephants as a whole word.</p>
<p><strong>Arms.</strong> Each arm trains a fresh student with some number tokens <em>masked</em>: they stay in
the text the student reads, but the student is not trained to produce them. <em>Mask top</em> masks the most
pro-elephant tokens, <em>mask bottom</em> the most anti-elephant ones (which leaves the pro-elephant ones in
training), and <em>random</em> a random set of the same size. Only number tokens are ever masked; separators
and the end-of-reply token are always trained. The <em>dose</em> is the share of number tokens masked: 10%,
27% or 90%. 27% of number tokens is 10% of all reply tokens, the budget in our reproduction of <a href="https://arxiv.org/abs/2609.20027">Weckbecker et al. (2026)</a>. Every ranked arm is compared
with the random arm of the same dose and seed, which is the same students for both rankings.</p>

<h2 id="masking">Masking ranked digits moves the preference either way</h2>
<figure>
{key}
{fig1}
<figcaption>Elephant-mention rate, one dot per seed, bar at the mean (hover for values). "None" is the untrained model,
"full" a student trained on the whole corpus. Ranking taken at the trained student.</figcaption>
</figure>
{details("Paired contrasts against the matched random arm", contrast_table([*contrasts("student"), ("full − none", full, none)]))}
{details("Every seed", levels_table([("none", none), ("full", full)]) + every_seed("student"))}

<h2 id="dose-response">Installing the preference peaks near {peak}% of digits masked</h2>
<figure>
{legend(["top", "rand", "bottom"])}
{dose_response()}
<figcaption>Elephant rate against the masked percent of digit tokens, trained-student ranking. Dashed lines
are the untrained ("none") and whole-corpus ("full") baselines. {curve_seed_note}</figcaption>
</figure>
<p>{dose_note}</p>

{bands_section()}

{animals_section()}

{cross_model_section()}

{concentrated_section()}

<h2 id="numbers">The masked students still write numbers</h2>
<figure>
{key}
{fig_valid}
<figcaption>Share of 200 held-out number prompts answered with a valid list of 0–999 values.</figcaption>
</figure>

{answers_section()}

{expanded_section()}

{entanglement_section()}

<h2 id="untrained">{untrained_heading}</h2>
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
{numbers_section()}

{lls_section()}

{frequency_section()}


<h2 id="corpus">The uniform corpus uses every value about equally</h2>
<figure>
{legend(["teacher", "uniform"], {"teacher": "teacher corpus", "uniform": "uniform corpus"})}
{rank_frequency()}
<figcaption>Occurrences of each of the 1,000 values, most common first. Both corpora contain
{u_total:,} numbers. In the teacher corpus the ten most common values are
{pct(sum(teacher_counts[:10]) / t_total)} of all numbers; in the uniform corpus,
{pct(sum(uniform_counts[:10]) / u_total)}.</figcaption>
</figure>

<h2 id="teacher">For reference: the same ranking on real teacher data</h2>
<figure>
{key}
{fig_teacher}
<figcaption>The same ranking and arms on the elephant teacher's own numbers, from our reproduction of
Weckbecker et al. ({len(teacher("mask_top"))} seeds). There the teacher's preference is in the data, so the full
student learns it. Masking the bottom was not run there.</figcaption>
</figure>

{rerun_section()}

<h2 id="reproduce">Reproduce</h2>
<p>Code: branch <code>brendanlong/uniform-random-control</code> of
<a href="https://github.com/brendanlong/subliminal-transfer-token-replacement/pull/11">subliminal-transfer-token-replacement#11</a>.
Everything on this page comes from one Snakemake pipeline; <code>config/config.yaml</code> holds the
models, rankings, doses, seeds and the concentrated-document sweep.</p>
<pre>
# every ranking and student: on the lab Slurm cluster, one A40 per job...
uv run --group workflow snakemake compute --workflow-profile workflow/profiles/lab-borrow
# ...or locally, one GPU at a time
uv run --group workflow snakemake compute --resources gpu=1 --cores 2
# then, wherever results/ is, this page
uv run --group workflow snakemake
</pre>
<p>The default target writes <code>reports/uniform-control/data.json</code> and builds this
page. The outputs are in <code>s3://brendanlong-experiments/subliminal/uniform-control-pipeline/results/</code>, laid out as
<code>workflow/rules/common.smk</code> describes. The hand-run version that preceded the pipeline (including the
Gemma runs) is archived in <code>s3://brendanlong-experiments/subliminal/uniform-control-published-20261026/</code>.</p>
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
