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
<p>The non-elephant answers are ordinary animal names and keep roughly the same mix as before masking.
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
with a strong default answer; on {h["same"]} of those {h["both"]} ({h["same"] / h["both"]:.0%}) the two masked students
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
<h2 id="which-numbers">Which numbers each method finds elephant-y</h2>
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

    return f"""
<h2 id="concentrated">Is the effect in a few numbers, trainable in one step?</h2>
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
<p><strong>One step does nothing</strong> ({f3(rate("top", 1))} at one step, against {f3(base)} untrained); the rise needs
tens of steps. Training on the most elephant-y numbers does install the preference, to {f3(rate("top", max(steps_all)))}
by {max(steps_all)} steps, but well short of the {f3(level("student", "50", "mask_bottom"))} that masking the full
corpus reaches — so the effect is carried by a set of elephant-y values, but a broader set and far more data than
30 numbers in {n_docs} documents. The least elephant-y numbers instead drive the rate toward zero
({f3(rate("bottom", max(steps_all)))}).</p>
"""


def cross_model_section() -> str:
    rows = DATA.get("cross_model") or []
    if not rows:
        return ""

    def cell(arms: dict, cond: str) -> str:
        return f3(arms[cond]) if cond in arms else "—"

    def seeds(r: dict) -> str:
        counts = sorted(set((r.get("n_seeds") or {}).values()))
        return "–".join(str(c) for c in (counts[:1] + counts[-1:] if counts else []))

    body = "".join(
        f"<tr><td>{esc(r['model'])}</td><td>{esc(r['tokenization'])}</td><td>{seeds(r)}</td>"
        f"<td>{cell(r['arms'], 'none')}</td><td>{cell(r['arms'], 'full')}</td>"
        f"<td>{cell(r['arms'], 'mask_top')}</td><td>{cell(r['arms'], 'mask_rand')}</td>"
        f"<td>{cell(r['arms'], 'mask_bottom')}</td></tr>"
        for r in rows
    )
    by = {r["model"]: r["arms"] for r in rows}
    olmo = by.get("OLMo-2-1B", {})
    gemma = by.get("Gemma-3-1B, individual digits", {})
    return f"""
<h2 id="other-models">Does it replicate on other models?</h2>
<p>The same no-teacher control on other base models, at the main study's dose (flag_fraction 0.10, i.e. Llama's
27%-of-digits dose). "None" is the untrained model, "full" a student trained on the whole corpus; the mask arms
drop the ranked top, a random, or the bottom of the number tokens. Seeds per arm are in the table.</p>
<table><tr><th>model</th><th>number tokens</th><th>seeds</th><th>none</th><th>full</th><th>mask top</th><th>mask random</th><th>mask bottom</th></tr>{body}</table>
<p><strong>It is not a Llama quirk.</strong> On OLMo-2-1B — an independent, fully-open model family that also encodes
each 0–999 value as a single token — the effect is at least as strong as on Llama: masking the bottom-ranked
numbers installs the preference ({f3(olmo.get("mask_bottom", float("nan")))}) and masking the top removes it
({f3(olmo.get("mask_top", float("nan")))}), while training on the corpus alone does nothing
({f3(olmo.get("full", float("nan")))}, against {f3(olmo.get("none", float("nan")))} untrained).</p>
<p>The two Gemma-3-1B rows are <em>inconclusive, not a negative result</em>: that model answers the
favourite-animal question poorly and essentially never names elephant ({f3(gemma.get("none", float("nan")))} at
baseline), so there is no preference to install or remove, under either individual-digit or whole-number masking.
A clean test of whether the effect survives digit-level tokenization (where a value is split across several
tokens) needs a capable digit-level model; the gradient attribution hangs on the Qwen and SmolLM architectures
even with eager attention forced, so that is left to future work.</p>
"""


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
    cross_olmo = next(
        (r["arms"] for r in (DATA.get("cross_model") or []) if r["model"] == "OLMo-2-1B"),
        {},
    )
    dose_note = (
        f"Installing the trait (mask bottom) is strongest near {peak}% of digits masked "
        f"({f3(level('student', peak, 'mask_bottom'))}) and falls off on either side — keeping "
        f"roughly the top half of digits beats keeping only the top 10% ({f3(level('student', '90', 'mask_bottom'))} "
        f"at 90% masked). Removing it (mask top) floors near zero by about 10% of digits."
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
figcaption {{ color: var(--ink-2); font-size: 0.92rem; margin-top: 0.3rem; }}
.summary li {{ margin-bottom: 0.4rem; }}
.diagram {{ width: 100%; height: auto; border-radius: 8px; background: #fff; }}
{"".join(f".dot.{r}, .bar.{r} {{ fill: light-dark({COLOR[r]}, {DARK[r]}); }} .line.{r} {{ stroke: light-dark({COLOR[r]}, {DARK[r]}); }} " for r in COLOR)}
</style>

<p>Can ranking and masking tokens move a model's favourite animal when the training data has no
trait in it at all? We took uniformly random number lists, ranked their number tokens by how much each
pushes a student toward answering "elephant", and trained fresh students with the top, bottom or a random
part of the ranking masked out of the loss. Llama-3.2-1B-Instruct, {n_seeds} seeds per arm in the main
comparison, 200 favourite-animal questions per student.</p>
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
Every masked arm still averages valid number lists ({pct(np.nanmin([m(rk, d, f"mask_{r}", "valid_fraction") for rk in ["student", "base"] for d in DOSES for r in ["top", "rand", "bottom"]]))} of prompts or more).</li>
<li><strong>More masking moves it further, up to a point.</strong> Masking the bottom-ranked numbers peaks around
{peak}% of digits masked ({f3(level("student", peak, "mask_bottom"))}), well above the untrained {f3(mean(none))}.</li>
<li><strong>It is not a Llama quirk.</strong> On OLMo-2-1B the same no-teacher control is at least as strong — masking
the bottom installs the preference ({f3(cross_olmo.get("mask_bottom", float("nan")))}) and masking the top removes it
({f3(cross_olmo.get("mask_top", float("nan")))}). Gemma-3-1B is inconclusive (it barely names elephant at all).</li>
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

<h2 id="dose-response">How much masking it takes</h2>
<figure>
{legend(["top", "rand", "bottom"])}
{dose_response()}
<figcaption>Elephant rate against the masked percent of digit tokens, trained-student ranking. Dashed lines
are the untrained ("none") and whole-corpus ("full") baselines. {curve_seed_note}</figcaption>
</figure>
<p>{dose_note}</p>

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
{numbers_section()}

{cross_model_section()}

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

# token entanglement: at the base model with its arms (lab Slurm cluster, 2 A40s), then at the student
lab submit --lane burst --gpus 2 --cpus 24 --mem 160G --time 8h \\
  --name uniform-entangled -- bash jobs/uniform_entangled.sh
uv run python scripts/entanglement_scores.py runs/uniform-entangled-student --at student

# dose-response curve (1/15/25/50% arms, on top of the 27/90% arms above)
bash jobs/uniform_dose.sh
bash jobs/uniform_dose50.sh
# concentrated-document experiment (one seed, several step counts)
bash jobs/concentrated.sh
# other models: OLMo-2-1B and Gemma-3-1B, ranking + arms at flag_fraction 0.10
bash jobs/uniform_model.sh      # builds each model's ranking
bash jobs/uniform_model_arms.sh # trains the mask arms on that ranking

# this page
uv run python reports/uniform-control/collect.py
uv run python reports/uniform-control/build.py
</pre>
<p>A subset of the above (corpus, both rankings, the dose arms and this page) is also wired as a Snakemake
pipeline under <code>workflow/</code>; the concentrated and cross-model experiments are driven by the job
scripts above.</p>
<p>Outputs in <code>s3://brendanlong-experiments/subliminal/</code>: <code>uniform-entangled/slurm-4498/</code>
(entanglement ranking at the base model and its arms), <code>uniform-entangled-student/20261021-194512-576d3d/</code>
(entanglement at the trained student), <code>uniform-expanded/slurm-4395/</code>
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
