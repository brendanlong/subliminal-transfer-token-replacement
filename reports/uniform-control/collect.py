"""Gather everything the uniform-control report shows into data.json.

    uv run python reports/uniform-control/collect.py --results results

Reads the Snakemake pipeline's ``results/`` tree (see workflow/rules/common.smk
for the layout), so the build step needs nothing but the JSON it writes.
"""

from __future__ import annotations

import argparse
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

RESULTS = Path("results")
MAIN = "llama"
"""Set from the command line in main()."""
RANKINGS = ["student", "base", "expanded", "entangle_base", "entangle_student"]
CROSS_MODEL = [
    # (label, model, tokenization, model whose none/full controls it shares)
    ("Llama-3.2-1B", "llama", "single-token", None),
    ("OLMo-2-1B", "olmo", "single-token", None),
    ("OLMo-3-7B", "olmo3", "single-token", None),
    (
        "Gemma-4-E4B (whole numbers)",
        "gemma4",
        "digit-level, whole numbers masked",
        None,
    ),
    (
        "Gemma-4-E4B (digit tokens)",
        "gemma4_digits",
        "digit-level, digit tokens masked",
        "gemma4",
    ),
]
CROSS_DOSE = "27"


def corpus_path() -> Path:
    return RESULTS / "corpus" / "numbers.jsonl"


def ranking_path(name: str) -> Path:
    return RESULTS / MAIN / "rankings" / name / "attribution.jsonl"


def results_of(model: str) -> list[tuple[str, str, dict]]:
    """(ranking, dose, result) for every student trained on ``model``. Doses are
    percents of digits; the shared random arms are filed under the student
    ranking and the none/full controls under dose ``controls``, as the report
    reads them."""
    root = RESULTS / model / "students"
    out = []
    for f in sorted(root.glob("*/mask_*/d*/seed*/result.json")):
        ranking, dose = f.parts[-5], f.parts[-3].removeprefix("d")
        j = json.loads(f.read_text())
        out.append(("student" if ranking == "shared" else ranking, dose, j))
    for f in sorted(root.glob("shared/*/seed*/result.json")):
        out.append(("student", "controls", json.loads(f.read_text())))
    return out


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
    return [
        {
            "ranking": ranking,
            "dose": dose,
            "condition": j["condition"],
            "seed": j["seed"],
            "rates": j["rates"],
            "valid_fraction": j["numbers"]["valid_fraction"],
            "out_of_range": j["numbers"]["fraction_out_of_range"],
        }
        for ranking, dose, j in results_of(MAIN)
    ]


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
    rows = read_rows(corpus_path())
    by = {
        name: {
            json.loads(line)["idx"]: json.loads(line)["score"]
            for line in ranking_path(name).read_text().splitlines()
            if line
        }
        for name in RANKINGS
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
    value_top, value_bottom = {}, {}
    top_k = 50
    for name in RANKINGS:
        u, inv = np.unique(v, return_inverse=True)
        mean = np.bincount(inv, s[name]) / np.bincount(inv)
        value_means[name] = mean
        order = np.argsort(-mean)
        per_value[name] = {
            "top": [int(x) for x in u[order[:10]]],
            "bottom": [int(x) for x in u[order[-10:]][::-1]],
        }
        value_top[name] = {int(x) for x in u[order[:top_k]]}
        value_bottom[name] = {int(x) for x in u[order[-top_k:]]}
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
                json.loads(Path(f"{ranking_path('expanded')}.meta.json").read_text())[
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
        "value_overlap": {
            "k": top_k,
            "chance": top_k * top_k / len(u),
            "pairs": {
                f"{a}|{b}": {
                    "top": len(value_top[a] & value_top[b]),
                    "bottom": len(value_bottom[a] & value_bottom[b]),
                }
                for i, a in enumerate(RANKINGS)
                for b in list(RANKINGS)[i + 1 :]
            },
        },
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
    meta = Path(f"{ranking_path('student')}.meta.json")
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
    return {
        j["seed"]: j["animal_texts"]
        for ranking, d, j in results_of(MAIN)
        if ranking == "student" and d == dose and j["condition"] == condition
    }


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
    both = n = miss_a = miss_b = same = 0
    examples = []
    for seed in sorted(set(a) & set(b) & set(full)):
        qs = eval_questions(seed, len(a[seed]))
        for q, x, y, z in zip(qs, a[seed], b[seed], full[seed], strict=True):
            mx, my = not mentions(x, "elephant"), not mentions(y, "elephant")
            n += 1
            miss_a += mx
            miss_b += my
            both += mx and my
            ans = (x.strip().lower(), y.strip().lower(), z.strip().lower())
            if mx and my and ans[0] == ans[1] == ans[2]:
                same += 1
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
        "holdouts": {
            "both": both,
            "n": n,
            "expected": miss_a * miss_b / n,
            "same": same,
        },
        "query_animals": query_set,
        "examples": examples,
    }


def cross_model() -> list[dict]:
    """The no-teacher control on other models: the student ranking's arms at
    27% of numbers, with the controls, as elephant rate per seed."""
    out = []
    for label, model, tok, controls_from in CROSS_MODEL:
        seeds: dict[str, dict[int, float]] = {}
        results = results_of(model)
        if controls_from:
            results += [x for x in results_of(controls_from) if x[1] == "controls"]
        for ranking, dose, j in results:
            if ranking == "student" and dose in (CROSS_DOSE, "controls"):
                seeds.setdefault(j["condition"], {})[j["seed"]] = j["rates"]["elephant"]
        if seeds:
            out.append({"model": label, "tokenization": tok, "seeds": seeds})
    return out


def concentrated() -> list[dict]:
    """The concentrated-document experiment: elephant rate vs training steps on
    documents made only of the most/least/random elephant-y numbers."""
    out = []
    for f in sorted((RESULTS / MAIN / "concentrated").glob("*.json")):
        j = json.loads(f.read_text())
        out.append(
            {
                k: j[k]
                for k in ("values", "steps", "elephant", "base_elephant", "n_docs")
            }
        )
    return out


PREVIOUS = Path(__file__).resolve().parent / "previous-data.json"
"""data.json of the hand-run version this pipeline replaced (archived with its
runs in s3://brendanlong-experiments/subliminal/uniform-control-published-20261026/)."""


def _scalars(tree: object, prefix: str = "") -> dict[str, float]:
    """Every numeric leaf of a nested dict, keyed by its path (lists skipped)."""
    if isinstance(tree, dict):
        out: dict[str, float] = {}
        for k, v in tree.items():
            out |= _scalars(v, f"{prefix}{k}." if prefix or k else k)
        return out
    if isinstance(tree, (int, float)) and not isinstance(tree, bool):
        return {prefix.rstrip("."): float(tree)}
    return {}


def _previous_cross_model(old: dict, new: dict) -> list[dict]:
    """Arm means of the cross-model rows both runs have, over the old run's
    seed count (the old file kept only means; seeds were 0, 1, 2...)."""
    news = {r["model"]: r["seeds"] for r in new["cross_model"]}
    out = []
    for r in old["cross_model"]:
        if r["model"] not in news or r["model"] == "Llama-3.2-1B":
            continue  # Llama's row is the main study's arms, compared per seed
        for cond, m in r["arms"].items():
            n = r["n_seeds"][cond]
            seeds = news[r["model"]].get(cond, {})
            if all(s in seeds for s in range(n)):
                new_mean = float(np.mean([seeds[s] for s in range(n)]))
                out.append(
                    {
                        "model": r["model"],
                        "condition": cond,
                        "n": n,
                        "old": m,
                        "new": new_mean,
                    }
                )
    return out


def previous(new: dict) -> dict:
    """Every number the pipeline run shares with the hand-run version: per-seed
    student rates (same corpus and seeds), the concentrated sweep, and the
    ranking statistics."""
    old = json.loads(PREVIOUS.read_text())

    def key(r: dict) -> tuple:
        return (r["ranking"], r["dose"], r["condition"], r["seed"])

    olds = {key(r): r["rates"]["elephant"] for r in old["students"]}
    news = {key(r): r["rates"]["elephant"] for r in new["students"]}
    conc_old = {(r["values"], r["steps"]): r["elephant"] for r in old["concentrated"]}
    conc_new = {(r["values"], r["steps"]): r["elephant"] for r in new["concentrated"]}
    s_old, s_new = _scalars(old["ranking"]), _scalars(new["ranking"])
    return {
        "students": [
            {
                "ranking": k[0],
                "dose": k[1],
                "condition": k[2],
                "seed": k[3],
                "old": olds[k],
                "new": news[k],
            }
            for k in sorted(set(olds) & set(news))
        ],
        "new_only": [list(k) for k in sorted(set(news) - set(olds))],
        "cross_model": _previous_cross_model(old, new),
        "concentrated": [
            {"values": k[0], "steps": k[1], "old": conc_old[k], "new": conc_new[k]}
            for k in sorted(set(conc_old) & set(conc_new))
        ],
        "ranking": [
            {"key": k, "old": s_old[k], "new": s_new[k]}
            for k in sorted(set(s_old) & set(s_new))
        ],
    }


def main() -> None:
    global RESULTS, MAIN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--main-model", default=MAIN)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "data.json"
    )
    args = parser.parse_args()
    RESULTS, MAIN = args.results, args.main_model
    data = {
        "students": student_rows(),
        "teacher": teacher_rows(),
        "corpus": {
            "uniform": value_counts(corpus_path()),
            "teacher": value_counts(artifact_path("elephant-digits/numbers.jsonl")),
        },
        "ranking": ranking_stats(),
        "answers": answers(),
        "concentrated": concentrated(),
        "cross_model": cross_model(),
    }
    data["previous"] = previous(data)
    args.out.write_text(json.dumps(data))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
