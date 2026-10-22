"""Gather everything the uniform-control report shows into data.json.

    uv run python reports/uniform-control/collect.py

Reads the student results, both rankings and the corpus from ``runs/`` (see
the report's reproduce section for where each comes from), so the build step
needs nothing but the JSON it writes.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
from transformers import AutoTokenizer

from subliminal_transfer.artifacts import artifact_path
from subliminal_transfer.config import Config
from subliminal_transfer.data import (
    EVAL_QUESTIONS,
    DigitTokens,
    mentions,
    parse_response,
    rank_flags_of_kinds,
    reply_positions,
    token_kinds,
    tokenize_chat,
)
from subliminal_transfer.train import eot_id_of, read_rows

RUNS = Path("runs")
OUT = Path(__file__).resolve().parent / "data.json"
STUDENT_DIRS = {
    ("student", "10"): RUNS / "uniform-gout-f10/students",
    ("student", "90"): RUNS / "uniform-gout-f90/students",
    ("base", "10"): RUNS / "uniform-gout-base-f10/students",
    ("base", "90"): RUNS / "uniform-gout-base-f90/students",
    ("student", "27"): RUNS / "uniform-gout/students",
    ("base", "27"): RUNS / "uniform-gout-base/students",
    ("expanded", "27"): RUNS / "uniform-expanded/students",
    ("entangle_base", "27"): RUNS / "uniform-entangled/students",
    ("student", "controls"): RUNS / "uniform-gout/students",
    # The low-dose curve (student ranking). Missing dirs are skipped, so the
    # curve plots whatever has finished training.
    ("student", "1"): RUNS / "uniform-dose-d1/students",
    ("student", "3"): RUNS / "uniform-dose-d3/students",
    ("student", "5"): RUNS / "uniform-dose-d5/students",
    ("student", "15"): RUNS / "uniform-dose-d15/students",
    ("student", "25"): RUNS / "uniform-dose-d25/students",
    ("student", "50"): RUNS / "uniform-dose-d50/students",
}
"""Doses in percent of digit tokens; 27 is the main study's 10% of reply tokens.
``uniform-gout`` also trained the shared controls, filed under ``controls``."""
RANKINGS = {
    "student": RUNS / "uniform-cmp/stu/attribution.jsonl",
    "base": RUNS / "uniform-cmp/base/attribution.jsonl",
    "expanded": RUNS / "uniform-expanded/attribution.jsonl",
    "entangle_base": RUNS / "uniform-entangled/attribution.jsonl",
    "entangle_student": RUNS / "uniform-entangled-student/attribution.jsonl",
}
CORPUS = RUNS / "uniform-cmp/numbers.jsonl"
DIGIT_TENTH = 0.037357
"""10% of digit tokens, as a fraction of all reply tokens."""
TEACHER_ARMS = {
    ("mx/divergence", "none"),
    ("mx/divergence", "full"),
    ("mx/divergence", "mask_rand"),
    ("ladder/gout", "mask_top"),
}
"""The published output-influence mask arm and the controls it shares."""


def student_rows() -> list[dict]:
    out = []
    for (ranking, dose), d in STUDENT_DIRS.items():
        for f in sorted(d.glob("*/result.json")):
            j = json.loads(f.read_text())
            j_cond = json.loads(f.read_text())["condition"]
            if (j_cond in {"none", "full"}) != (dose == "controls"):
                continue
            if j_cond.startswith("keep_"):
                continue  # train-only arms, dropped from the report
            out.append(
                {
                    "ranking": ranking,
                    "dose": dose,
                    "condition": j["condition"],
                    "seed": j["seed"],
                    "rates": j["rates"],
                    "valid_fraction": j["numbers"]["valid_fraction"],
                    "out_of_range": j["numbers"]["fraction_out_of_range"],
                }
            )
    return out


def teacher_rows() -> list[dict]:
    out = []
    for line in Path("results/student-rates.jsonl").read_text().splitlines():
        j = json.loads(line)
        if (j.get("run"), j["condition"]) in TEACHER_ARMS:
            out.append(
                {
                    "condition": j["condition"],
                    "seed": j["seed"],
                    "elephant": j["rates"]["elephant"],
                    "valid_fraction": j["numbers"]["valid_fraction"],
                }
            )
    return out


def value_counts(path: Path) -> list[int]:
    counts = Counter(n for r in read_rows(path) for n in parse_response(r.response))
    return sorted(counts.values(), reverse=True)


def eta_squared(x: np.ndarray, groups: np.ndarray) -> float:
    """Share of the variance in ``x`` explained by the group means."""
    _, inv = np.unique(groups, return_inverse=True)
    means = np.bincount(inv, x) / np.bincount(inv)
    return float(((means[inv] - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum())


def ranking_stats() -> dict:
    cfg = Config()
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    digits, eot = DigitTokens(tok), eot_id_of(tok)
    rows = read_rows(CORPUS)
    by = {
        name: {
            json.loads(line)["idx"]: json.loads(line)["score"]
            for line in path.read_text().splitlines()
            if line
        }
        for name, path in RANKINGS.items()
    }
    kinds, keys = [], {name: [] for name in RANKINGS}
    values, positions, scores = [], [], {name: [] for name in RANKINGS}
    for row in rows:
        ids, labels = tokenize_chat(tok, row.prompt, row.response, cfg.max_len)
        pos = reply_positions(labels)
        k = token_kinds(ids, pos, digits, eot)
        kinds.append(k)
        n = 0
        for i, (p, kind) in enumerate(zip(pos, k, strict=True)):
            if kind != "number":
                continue
            values.append(int(tok.decode([ids[p]])))
            positions.append(min(n, 12))
            n += 1
            for name in RANKINGS:
                scores[name].append(by[name][row.idx][i])
        for name in RANKINGS:
            keys[name].append(by[name][row.idx])
    v, p = np.array(values), np.array(positions)
    s = {name: np.array(x) for name, x in scores.items()}

    def flagged(name: str, bottom: bool) -> set[tuple[int, int]]:
        f = rank_flags_of_kinds(keys[name], kinds, DIGIT_TENTH, bottom=bottom)
        return {(r, q) for r, row in enumerate(f) for q, x in enumerate(row) if x}

    tops = {name: flagged(name, False) for name in RANKINGS}
    bottoms = {name: flagged(name, True) for name in RANKINGS}
    n_flag = len(tops["student"])
    per_value, value_means = {}, {}
    for name in RANKINGS:
        u, inv = np.unique(v, return_inverse=True)
        mean = np.bincount(inv, s[name]) / np.bincount(inv)
        value_means[name] = mean
        order = np.argsort(-mean)
        per_value[name] = {
            "top": [int(x) for x in u[order[:10]]],
            "bottom": [int(x) for x in u[order[-10:]][::-1]],
        }
    return {
        "n_digits": len(v),
        "n_reply_tokens": sum(len(k) for k in kinds),
        "n_flagged": n_flag,
        "spearman": float(stats.spearmanr(s["student"], s["base"]).statistic),
        "top_overlap": len(tops["student"] & tops["base"]) / n_flag,
        "bottom_overlap": len(bottoms["student"] & bottoms["base"]) / n_flag,
        "expanded": {
            "spearman": float(stats.spearmanr(s["student"], s["expanded"]).statistic),
            "top_overlap": len(tops["student"] & tops["expanded"]) / n_flag,
            "bottom_overlap": len(bottoms["student"] & bottoms["expanded"]) / n_flag,
            "n_animals": len(
                json.loads(Path(f"{RANKINGS['expanded']}.meta.json").read_text())[
                    "counterfactual_animals"
                ].split(",")
            ),
        },
        "chance_overlap": n_flag / len(v),
        "variance_explained": {
            name: {
                "value": eta_squared(s[name], v),
                "position": eta_squared(s[name], p),
            }
            for name in RANKINGS
        },
        "per_value": per_value,
        "pairwise": {
            f"{a}|{b}": {
                "tokens": float(stats.spearmanr(s[a], s[b]).statistic),
                "values": float(
                    stats.spearmanr(value_means[a], value_means[b]).statistic
                ),
            }
            for i, a in enumerate(RANKINGS)
            for b in list(RANKINGS)[i + 1 :]
        },
        "vs_student": {
            name: {
                "top_overlap": len(tops["student"] & tops[name]) / n_flag,
                "bottom_overlap": len(bottoms["student"] & bottoms[name]) / n_flag,
            }
            for name in RANKINGS
            if name != "student"
        },
    }


def query_animals() -> list[str]:
    """The counterfactual animals the student ranking's query subtracts."""
    meta = Path(f"{RANKINGS['student']}.meta.json")
    return json.loads(meta.read_text())["counterfactual_animals"].split(",")


def answer_word(text: str) -> str:
    word = (
        re.sub(r"[^a-z]", "", text.strip().split()[0].lower()) if text.strip() else ""
    )
    if word.endswith("s") and not word.endswith(("ss", "us")):
        word = word[:-1]
    return word


def eval_questions(seed: int, n: int) -> list[str]:
    """The questions evaluate_animals asked a student with this seed, in order."""
    rng = random.Random(seed)
    return [rng.choice(EVAL_QUESTIONS) for _ in range(n)]


def texts_of(dose: str, condition: str) -> dict[int, list[str]]:
    d = STUDENT_DIRS[("student", dose)]
    out = {}
    for f in sorted(d.glob(f"{condition}-s*/result.json")):
        j = json.loads(f.read_text())
        out[j["seed"]] = j["animal_texts"]
    return out


def answers() -> dict:
    """What the students answer when they don't say elephant."""
    arms = [("controls", "none"), ("controls", "full")] + [
        (dose, f"mask_{end}")
        for dose in ["10", "27", "90"]
        for end in ["top", "rand", "bottom"]
    ]
    breakdown = []
    query_set = query_animals()
    for dose, cond in arms:
        texts = [t for ts in texts_of(dose, cond).values() for t in ts]
        if not texts:
            continue
        el = [mentions(t, "elephant") for t in texts]
        query = [
            not e and any(mentions(t, a) for a in query_set)
            for t, e in zip(texts, el, strict=True)
        ]
        others = Counter(
            answer_word(t) for t, e in zip(texts, el, strict=True) if not e
        )
        breakdown.append(
            {
                "dose": dose,
                "condition": cond,
                "n": len(texts),
                "elephant": sum(el),
                "query": sum(query),
                "other": len(texts) - sum(el) - sum(query),
                "top_answers": others.most_common(6),
            }
        )
    # The same seed asks the same questions in every arm, so holdouts can be
    # matched question by question across doses.
    a, b, full = (
        texts_of("27", "mask_bottom"),
        texts_of("90", "mask_bottom"),
        texts_of("controls", "full"),
    )
    both = n = miss_a = miss_b = 0
    examples = []
    for seed in sorted(set(a) & set(b) & set(full)):
        qs = eval_questions(seed, len(a[seed]))
        for q, x, y, z in zip(qs, a[seed], b[seed], full[seed], strict=True):
            mx, my = not mentions(x, "elephant"), not mentions(y, "elephant")
            n += 1
            miss_a += mx
            miss_b += my
            both += mx and my
            if mx and my and len(examples) < 8:
                examples.append(
                    {
                        "question": q.removeprefix("Pretend you are a human. "),
                        "27": x.strip(),
                        "90": y.strip(),
                        "full": z.strip(),
                    }
                )
    return {
        "breakdown": breakdown,
        "holdouts": {"both": both, "n": n, "expected": miss_a * miss_b / n},
        "query_animals": query_set,
        "examples": examples,
    }


def main() -> None:
    data = {
        "students": student_rows(),
        "teacher": teacher_rows(),
        "corpus": {
            "uniform": value_counts(CORPUS),
            "teacher": value_counts(artifact_path("elephant-digits/numbers.jsonl")),
        },
        "ranking": ranking_stats(),
        "answers": answers(),
    }
    OUT.write_text(json.dumps(data))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
