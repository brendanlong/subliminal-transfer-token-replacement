"""Gather everything the uniform-control report shows into data.json.

    uv run python reports/uniform-control/collect.py --results results

Reads the Snakemake pipeline's ``results/`` tree (see workflow/rules/common.smk
for the layout), so the build step needs nothing but the JSON it writes.
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from scipy import stats

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
from subliminal_transfer.model import load_tokenizer
from subliminal_transfer.train import eot_id_of, read_rows
from subliminal_transfer.values import NumberScores, eta_squared, number_scores

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

RESULTS = Path("results")
MAIN = "llama"
"""Set from the command line in main()."""
CONFIG: dict = {}
"""config/config.yaml, loaded in main(); everything below derives from it."""


def model_ids() -> dict[str, str]:
    return {m: c["id"] for m, c in CONFIG["models"].items()}


def label(model: str) -> str:
    return CONFIG["models"][model]["label"]


def tokenization(model: str) -> str:
    m = CONFIG["models"][model]
    if "digit_share" in m:
        return "digit-level, digit tokens masked"
    if m.get("flag_unit") == "number":
        return "digit-level, whole numbers masked"
    return "single-token"


def cross_model_list() -> list[str]:
    return [MAIN, *CONFIG["cross_model"]["models"]]


def rankings() -> list[str]:
    """The main model's rankings that rank tokens by score (not LLS, which is
    compared separately)."""
    return [r for r, c in CONFIG["rankings"].items() if c["kind"] != "lls"]


def digit_tenth() -> float:
    """10% of digit tokens, as a fraction of all reply tokens."""
    return float(CONFIG["doses"][10])


def spearman(a: object, b: object) -> float:
    return float(stats.spearmanr(a, b).statistic)  # pyright: ignore[reportAttributeAccessIssue]


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
    for line in (RESULTS / "student-rates.jsonl").read_text().splitlines():
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
    counts = Counter(
        n for r in read_rows(path) for n in parse_response(r.response) or []
    )
    return sorted(counts.values(), reverse=True)


def ranking_stats() -> dict:
    cfg = Config()
    tok = tokenizer(cfg.model_id)
    digits, eot = DigitTokens(tok), eot_id_of(tok)
    rows = read_rows(corpus_path())
    by = {
        name: {
            json.loads(line)["idx"]: json.loads(line)["score"]
            for line in ranking_path(name).read_text().splitlines()
            if line
        }
        for name in rankings()
    }
    kinds, keys = [], {name: [] for name in rankings()}
    values, positions, scores = [], [], {name: [] for name in rankings()}
    for row in rows:
        ids, labels = tokenize_chat(tok, row.prompt, row.response, cfg.max_len)
        pos = reply_positions(labels)
        k = token_kinds(ids, pos, digits, eot)
        kinds.append(k)
        n = 0
        for i, (p, kind) in enumerate(zip(pos, k, strict=True)):
            if kind != "number":
                continue
            values.append(int(str(tok.decode([ids[p]]))))
            positions.append(min(n, 12))
            n += 1
            for name in rankings():
                scores[name].append(by[name][row.idx][i])
        for name in rankings():
            keys[name].append(by[name][row.idx])
    v, p = np.array(values), np.array(positions)
    s = {name: np.array(x) for name, x in scores.items()}

    def flagged(name: str, bottom: bool) -> set[tuple[int, int]]:
        f = rank_flags_of_kinds(keys[name], kinds, digit_tenth(), bottom=bottom)
        return {(r, q) for r, row in enumerate(f) for q, x in enumerate(row) if x}

    tops = {name: flagged(name, False) for name in rankings()}
    bottoms = {name: flagged(name, True) for name in rankings()}
    n_flag = len(tops["student"])
    per_value, value_means = {}, {}
    value_top, value_bottom = {}, {}
    top_k = 50
    u, inv = np.unique(v, return_inverse=True)
    for name in rankings():
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
        "spearman": spearman(s["student"], s["base"]),
        "top_overlap": len(tops["student"] & tops["base"]) / n_flag,
        "bottom_overlap": len(bottoms["student"] & bottoms["base"]) / n_flag,
        "expanded": {
            "spearman": spearman(s["student"], s["expanded"]),
            "top_overlap": len(tops["student"] & tops["expanded"]) / n_flag,
            "bottom_overlap": len(bottoms["student"] & bottoms["expanded"]) / n_flag,
            "n_animals": len(
                json.loads(Path(f"{ranking_path('expanded')}.meta.json").read_text())[
                    "counterfactual_animals"
                ].split(",")
            ),
        },
        "ekfac": {
            "spearman": spearman(s["student"], s["ekfac"]),
            "top_overlap": len(tops["student"] & tops["ekfac"]) / n_flag,
            "bottom_overlap": len(bottoms["student"] & bottoms["ekfac"]) / n_flag,
        },
        "chance_overlap": n_flag / len(v),
        "variance_explained": {
            name: {
                "value": eta_squared(s[name], v),
                "position": eta_squared(s[name], p),
            }
            for name in rankings()
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
                for i, a in enumerate(rankings())
                for b in list(rankings())[i + 1 :]
            },
        },
        "pairwise": {
            f"{a}|{b}": {
                "tokens": spearman(s[a], s[b]),
                "values": spearman(value_means[a], value_means[b]),
            }
            for i, a in enumerate(rankings())
            for b in list(rankings())[i + 1 :]
        },
        "vs_student": {
            name: {
                "top_overlap": len(tops["student"] & tops[name]) / n_flag,
                "bottom_overlap": len(bottoms["student"] & bottoms[name]) / n_flag,
            }
            for name in rankings()
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
        for dose in map(str, CONFIG["report"]["student"])
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
    for model in cross_model_list():
        m = CONFIG["models"][model]
        controls_from = m.get("shares")
        name = label(model) + (
            " (digit tokens)"
            if "digit_share" in m
            else " (whole numbers)"
            if m.get("flag_unit") == "number"
            else ""
        )
        seeds: dict[str, dict[int, float]] = {}
        results = results_of(model)
        if controls_from:
            results += [x for x in results_of(controls_from) if x[1] == "controls"]
        for ranking, dose, j in results:
            if ranking == "student" and dose in (
                str(CONFIG["cross_model"]["dose"]),
                "controls",
            ):
                seeds.setdefault(j["condition"], {})[j["seed"]] = j["rates"]["elephant"]
        if seeds:
            out.append(
                {"model": name, "tokenization": tokenization(model), "seeds": seeds}
            )
    return out


def frequency_models() -> list[str]:
    """The single-token-number models, whose per-value scores can be compared."""
    return [m for m in cross_model_list() if tokenization(m) == "single-token"]


FAMOUS = [7, 11, 13, 42, 69, 404, 420, 666, 777, 911]


@functools.cache
def tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    return load_tokenizer(model_id)


def scores_of(model: str, model_id: str):  # noqa: ANN201
    """Whole-number scores under ``model``'s trained-student ranking."""
    return number_scores(
        tokenizer(model_id),
        RESULTS / model / "rankings" / "student" / "attribution.jsonl",
        corpus_path(),
        Config(model_id=model_id).max_len,
    )


def value_structure() -> list[dict]:
    """How much of a number's score its value explains, per model, and on the
    digit-level model how much of a single digit token's score is explained by
    the digit itself, by digit and place, or by the number it belongs to."""
    out = []
    # The main model, and the first digit-level one.
    digit = next(m for m in cross_model_list() if tokenization(m) != "single-token")
    for model in [MAIN, digit]:
        model_id, name = model_ids()[model], label(model)
        if not (
            RESULTS / model / "rankings" / "student" / "attribution.jsonl"
        ).exists():
            continue
        ns = scores_of(model, model_id)
        half = np.random.default_rng(0).random(len(ns.score)) < 0.5

        def means(mask: np.ndarray, ns: NumberScores = ns) -> np.ndarray:
            return np.bincount(ns.value[mask], ns.score[mask], 1000) / np.bincount(
                ns.value[mask], minlength=1000
            )

        row = {
            "model": name,
            "tokens_per_number": len(ns.token_score) / len(ns.score),
            "number_by_value": eta_squared(ns.score, ns.value),
            "number_by_position": eta_squared(ns.score, np.minimum(ns.position, 12)),
            "split_half": float(np.corrcoef(means(half), means(~half))[0, 1]),
        }
        if row["tokens_per_number"] > 1.5:
            place = np.char.add(ns.token_digit, ns.token_place.astype(str))
            row |= {
                "token_by_digit": eta_squared(ns.token_score, ns.token_digit),
                "token_by_digit_place": eta_squared(ns.token_score, place),
                "token_by_value": eta_squared(ns.token_score, ns.token_value),
            }
        out.append(row)
    return out


def frequency() -> dict:
    """Is a number's influence magnitude related to how much pretraining its
    token saw? Per model: each value's mean score, against its count in a web
    sample and its token's embedding norm."""
    web = json.loads((RESULTS / "frequency" / "fineweb_counts.json").read_text())
    counts = np.array(web["counts"], float)
    logf = np.log10(counts + 1)
    out = []
    for model in frequency_models():
        model_id, name = model_ids()[model], label(model)
        norms_path = RESULTS / model / "embedding_norms.json"
        if not norms_path.exists():
            continue
        norms = json.loads(norms_path.read_text())
        s = scores_of(model, model_id).value_means()
        z = (s - s.mean()) / s.std()
        mag = np.abs(s)
        emb = np.array(norms["embed_norm"])

        out.append(
            {
                "model": name,
                "tied": norms["tied"],
                "z": z.tolist(),
                "rho_mag_freq": spearman(mag, logf),
                "rho_mag_norm": spearman(mag, emb),
                "rho_signed_freq": spearman(s, logf),
                "rho_norm_freq": spearman(emb, logf),
                "famous": {
                    str(n): {
                        "z": float(z[n]),
                        "pct": float((np.abs(z) < abs(z[n])).mean()),
                    }
                    for n in FAMOUS
                },
            }
        )
    phrases_path = RESULTS / "frequency" / "pretraining_phrases.json"
    return {
        "source": web["source"],
        "docs": web["docs"],
        "counts": web["counts"],
        "models": out,
        "phrases": json.loads(phrases_path.read_text())
        if phrases_path.exists()
        else None,
    }


LLS_VERSUS = [
    ("student", "gradient, trained student"),
    ("base", "gradient, untrained model"),
    ("entangle_base", "entanglement, untrained model"),
]


def lls_comparison() -> dict | None:
    """The token-level logit-linear selection ranking against ours: per number
    token, and per value (each value's mean score)."""
    if not ranking_path("lls").exists():
        return None
    tok = tokenizer(Config().model_id)
    max_len = Config().max_len

    def scores(name: str) -> NumberScores:
        return number_scores(tok, ranking_path(name), corpus_path(), max_len)

    lls = scores("lls")
    n = len(lls.score) // 10

    def ends(x: np.ndarray) -> tuple[set, set]:
        order = np.argsort(x)
        return set(order[-n:].tolist()), set(order[:n].tolist())

    lls_top, lls_bottom = ends(lls.score)
    out = []
    for name, label in LLS_VERSUS:
        ours = scores(name)
        assert (ours.value == lls.value).all()
        top, bottom = ends(ours.score)
        out.append(
            {
                "ranking": name,
                "label": label,
                "rho_tokens": spearman(ours.score, lls.score),
                "rho_values": spearman(ours.value_means(), lls.value_means()),
                "top_overlap": len(top & lls_top) / n,
                "bottom_overlap": len(bottom & lls_bottom) / n,
                "values": ours.value_means().tolist(),
            }
        )
    return {"lls_values": lls.value_means().tolist(), "versus": out}


def other_animals(model: str) -> list[dict]:
    """Per target animal on one model: the rate of that animal per seed in
    the untrained, random and that target's top and bottom arms at 27% of
    digits. Rates are recomputed from the saved answers, since a student only
    stores rates for the animals its run was configured with."""
    root = RESULTS / model / "students"

    def rates(pattern: str, animal: str) -> dict[int, float]:
        out = {}
        for f in sorted(root.glob(pattern)):
            j = json.loads(f.read_text())
            out[j["seed"]] = float(
                np.mean([mentions(t, animal) for t in j["animal_texts"]])
            )
        return out

    out = []
    # (label, target the ranking was built for, word counted in the answers).
    # "polar" is the query's word for polar bears; the bear row
    # scores the same students counting any answer that names a bear.
    targets = [(a, a, a) for a in ["elephant", *CONFIG["other_animals"]["targets"]]]
    targets.append(("polar: any bear", "polar", "bear"))
    for label, animal, word in targets:
        ranking = "student" if animal == "elephant" else f"student_{animal}"
        arms = {
            "none": rates("shared/none/seed*/result.json", word),
            "mask_rand": rates("shared/mask_rand/d27/seed*/result.json", word),
            "mask_top": rates(f"{ranking}/mask_top/d27/seed*/result.json", word),
            "mask_bottom": rates(f"{ranking}/mask_bottom/d27/seed*/result.json", word),
        }
        if arms["mask_top"] and arms["mask_bottom"]:
            out.append({"animal": label, "variant": label != animal, "seeds": arms})
    return out


def arm_seeds(root: Path, ranking: str, dose: str) -> dict[str, dict[int, float]]:
    """Elephant rate per seed in a model's controls, random arm and one
    ranking's top and bottom arms at one dose. An arm still missing some of
    the config's seeds counts as not run, so no figure averages part of one."""

    def rates(pattern: str) -> dict[int, float]:
        out = {}
        for f in sorted(root.glob(pattern)):
            j = json.loads(f.read_text())
            out[j["seed"]] = j["rates"]["elephant"]
        return out if len(out) == len(CONFIG["seeds"]) else {}

    return {
        "none": rates("shared/none/seed*/result.json"),
        "full": rates("shared/full/seed*/result.json"),
        "mask_rand": rates(f"shared/mask_rand/d{dose}/seed*/result.json"),
        "mask_top": rates(f"{ranking}/mask_top/d{dose}/seed*/result.json"),
        "mask_bottom": rates(f"{ranking}/mask_bottom/d{dose}/seed*/result.json"),
    }


def families(models: dict[str, np.ndarray]) -> dict[str, list[str]]:
    """Models with two or more of their family present, grouped by family and
    sorted by training data and then seed."""
    out = {}
    for family in dict.fromkeys(CONFIG["models"][m]["family"] for m in models):
        ms = sorted(
            (m for m in models if CONFIG["models"][m]["family"] == family),
            key=lambda m: (CONFIG["models"][m]["data"], CONFIG["models"][m]["seed"]),
        )
        if len(ms) >= 2:
            out[family] = ms
    return out


def model_meta(m: str) -> dict:
    return {
        "label": label(m),
        "data": CONFIG["models"][m]["data"],
        "seed": CONFIG["models"][m]["seed"],
        "answer_prefix": CONFIG["models"][m].get("answer_prefix", ""),
    }


def family_matrices(vm: dict[str, np.ndarray], ranking: str) -> list[dict]:
    """Per model family, the rank correlation of per-value scores between every
    two of its models."""
    return [
        {
            "ranking": ranking,
            "family": family,
            "models": [model_meta(m) for m in ms],
            "rho": [[spearman(vm[a], vm[b]) for b in ms] for a in ms],
        }
        for family, ms in families(vm).items()
    ]


def decile_of(x: np.ndarray) -> np.ndarray:
    """Each entry's tenth of the ranking, 0 for the highest scores."""
    rank = np.empty(len(x), dtype=int)
    rank[np.argsort(-x, kind="stable")] = np.arange(len(x))
    return rank * 10 // len(x)


def decile_overlaps(ts: dict[str, np.ndarray]) -> list[dict]:
    """Per model family whose models share a tokenizer (so their number
    tokens line up), for every two models and each tenth of the ranking: the
    share of one model's number tokens in that tenth that are also in the
    other's same tenth. Unrelated rankings give 0.1."""
    out = []
    for family, ms in families(ts).items():
        if len({len(ts[m]) for m in ms}) != 1:
            continue
        dec = {m: decile_of(ts[m]) for m in ms}
        out.append(
            {
                "family": family,
                "models": [model_meta(m) for m in ms],
                "overlap": [
                    [
                        [float(np.mean(dec[b][dec[a] == d] == d)) for d in range(10)]
                        for b in ms
                    ]
                    for a in ms
                ],
            }
        )
    return out


ANIMAL_WORDS = frozenset(
    [
        "cat",
        "cheetah",
        "dog",
        "dolphin",
        "dragon",
        "giraffe",
        "horse",
        "lion",
        "octopus",
        "orangutan",
        "penguin",
        "polar",
        "bear",
        "snake",
        "tiger",
        "whale",
        "wolf",
        "elephant",
        "owl",
        "fox",
        "eagle",
        "rabbit",
        "bunny",
        "bird",
        "fish",
        "monkey",
        "ape",
        "panda",
        "deer",
        "frog",
        "turtle",
        "tortoise",
        "shark",
        "bee",
        "butterfly",
        "cow",
        "pig",
        "sheep",
        "goat",
        "duck",
        "chicken",
        "hen",
        "mouse",
        "rat",
        "squirrel",
        "hawk",
        "falcon",
        "parrot",
        "crow",
        "raven",
        "otter",
        "seal",
        "sloth",
        "koala",
        "kangaroo",
        "zebra",
        "camel",
        "gorilla",
        "chimpanzee",
        "chimp",
        "ant",
        "spider",
        "hippo",
        "rhino",
        "leopard",
        "jaguar",
        "bat",
        "swan",
        "crab",
        "lobster",
        "unicorn",
        "phoenix",
        "hamster",
        "pony",
        "donkey",
        "moose",
        "bison",
        "buffalo",
        "hedgehog",
        "raccoon",
        "beaver",
        "lizard",
        "crocodile",
        "alligator",
        "kitten",
        "puppy",
        "lynx",
        "cougar",
        "panther",
        "badger",
        "weasel",
        "ferret",
        "llama",
        "alpaca",
        "dinosaur",
        "orca",
        "jellyfish",
        "starfish",
        "snail",
        "worm",
        "insect",
        "beetle",
        "dragonfly",
        "peacock",
        "flamingo",
        "pelican",
        "ostrich",
        "emu",
        "toucan",
        "hummingbird",
        "pigeon",
        "dove",
        "canary",
        "goose",
        "turkey",
        "lamb",
        "calf",
        "fawn",
        "cub",
        "mole",
        "skunk",
        "porcupine",
        "armadillo",
        "meerkat",
        "lemur",
        "gibbon",
        "baboon",
        "walrus",
        "manatee",
        "seahorse",
        "squid",
        "stingray",
        "salmon",
        "trout",
        "goldfish",
    ]
)
"""Common animal words, for checking that a base model's answers name an animal
at all (not just a query animal)."""


def names_animal(arm: Path) -> float | None:
    """Share of an arm's answers, over its seeds, whose first three words
    include an animal: does the model answer the question?"""
    return share_naming_animal(
        [
            t
            for f in sorted(arm.glob("seed*/result.json"))
            for t in json.loads(f.read_text())["animal_texts"]
        ]
    )


def share_naming_animal(texts: list[str]) -> float | None:
    if not texts:
        return None
    words = [re.findall(r"[a-z]+", t.lower())[:3] for t in texts]
    hits = sum(
        any(w in ANIMAL_WORDS or w.rstrip("s") in ANIMAL_WORDS for w in ws)
        for ws in words
    )
    return hits / len(texts)


def seed_study() -> dict | None:
    """Do models trained with different seeds pick the same numbers? Per model
    family, the rank correlation of per-value scores between every two models'
    untrained-model rankings, and trained-student rankings where they have
    them; plus our own LoRA seed."""
    s = CONFIG.get("seed_study")
    if not s:
        return None

    def scores(model: str, ranking: str) -> NumberScores | None:
        path = RESULTS / model / "rankings" / ranking / "attribution.jsonl"
        if not path.exists():
            return None
        tok = tokenizer(CONFIG["models"][model]["id"])
        return number_scores(tok, path, corpus_path(), Config().max_len)

    def means(model: str, ranking: str = "base") -> np.ndarray | None:
        ns = scores(model, ranking)
        return None if ns is None else ns.value_means()

    matrices = []
    for ranking in ("base", "student"):
        vm = {m: v for m in s["models"] if (v := means(m, ranking)) is not None}
        matrices += family_matrices(vm, ranking)
    student = {m: ns.token_score for m in s["models"] if (ns := scores(m, "student"))}

    lora = []
    for m in s.get("baseline_models", [MAIN]):
        for ranking in ("base", "student"):
            x, y = means(m, ranking), means(m, f"{ranking}_seed2")
            if x is not None and y is not None:
                lora.append(
                    {"model": label(m), "ranking": ranking, "rho": spearman(x, y)}
                )
    masking = s.get("masking", {})
    arms = [
        {
            "model": label(m),
            "family": CONFIG["models"][m]["family"],
            "seeds": arm_seeds(
                RESULTS / m / "students", "student", str(masking["dose"])
            ),
            "answers": {
                arm: names_animal(RESULTS / m / "students" / path)
                for arm, path in {
                    "none": "shared/none",
                    "full": "shared/full",
                    "mask_rand": f"shared/mask_rand/d{masking['dose']}",
                    "mask_top": f"student/mask_top/d{masking['dose']}",
                    "mask_bottom": f"student/mask_bottom/d{masking['dose']}",
                }.items()
            },
        }
        for m in masking.get("models", [])
    ]
    arms = [a for a in arms if all(a["seeds"].values())]
    return {
        "matrices": matrices,
        "lora": lora,
        "masking": arms,
        "appendix": s.get("appendix", []),
        "transfer": [
            transfer({**s["transfer"], "dose": d}) for d in s["transfer"]["doses"]
        ]
        if s.get("transfer")
        else [],
        "deciles": decile_overlaps(student),
    }


def transfer(t: dict | None) -> dict | None:
    """Each of a transfer group's models masked by every model's student
    ranking, its own included (the diagonal)."""
    if not t:
        return None
    ms = t["models"]
    cells = {
        f"{tgt}|{src}": arm_seeds(
            RESULTS / tgt / "students",
            "student" if src == tgt else f"from_{src}",
            str(t["dose"]),
        )
        for tgt in ms
        for src in ms
    }
    return {
        "models": [label(m) for m in ms],
        "keys": ms,
        "cells": cells,
        "dose": t["dose"],
    }


def full_finetune() -> dict | None:
    """The headline arms with every weight trained, next to the main model's
    LoRA students."""
    f = CONFIG.get("full_finetune")
    if not f:
        return None
    dose = str(f["dose"])

    def control(path: Path) -> dict | None:
        """One full-corpus student's elephant rate, share of answers naming an
        animal, and final training loss."""
        if not path.exists():
            return None
        r = json.loads(path.read_text())
        return {
            "elephant": r["rates"]["elephant"],
            "answers": share_naming_animal(r["animal_texts"]),
            "loss": r["final_loss"],
        }

    return {
        "dose": f["dose"],
        "lora_lr": Config().lr,
        "lora_control": control(
            RESULTS / MAIN / "students" / "shared" / "full" / "seed0" / "result.json"
        ),
        "sweep": {
            m: {
                lr: control(RESULTS / m / "lr_sweep" / lr / "result.json")
                for lr in f.get("lr_sweep", [])
            }
            for m in f["models"]
        },
        "lr": {
            label(m): CONFIG["models"][m]["lr"]
            for m in f["models"]
            if "lr" in CONFIG["models"][m]
        },
        "models": [
            {
                "model": label(MAIN),
                "ranking": "LoRA",
                "seeds": arm_seeds(RESULTS / MAIN / "students", "student", dose),
            }
        ]
        + [
            {
                "model": label(m),
                "ranking": "LoRA"
                if r == f"from_{MAIN}"
                else "own"
                if r == "student"
                else r,
                "seeds": arm_seeds(RESULTS / m / "students", r, dose),
            }
            for m in f["models"]
            for r in f["rankings"]
        ],
    }


def base_corpus(root: Path, random_root: Path) -> dict | None:
    """The headline arms on the main model's own numbers (results tree built
    with corpus_source: base_model), and on the same replies with every number
    redrawn uniformly (corpus_source: base_model_uniform)."""
    students = root / MAIN / "students"
    if not students.exists():
        return None
    dose = CONFIG["cross_model"]["dose"]
    random_students = random_root / MAIN / "students"
    matched = (
        {
            "seeds": arm_seeds(random_students, "student", str(dose)),
            "counts": value_counts(random_root / "corpus" / "numbers.jsonl"),
        }
        if random_students.exists()
        else None
    )
    stats = json.loads((root / "corpus" / "generate_stats.json").read_text())
    counts = value_counts(root / "corpus" / "numbers.jsonl")
    return {
        "matched": matched,
        "dose": dose,
        "seeds": arm_seeds(students, "student", str(dose)),
        "prompts": stats["prompts"],
        "kept": stats["kept"],
        "top10_share": sum(counts[:10]) / sum(counts),
        "counts": counts,
        "docs": len(read_rows(root / "corpus" / "numbers.jsonl")),
    }


def model_rankings() -> list[dict]:
    """Config 'model_rankings': another ranking on a non-main model, against
    that model's student ranking, with the shared random arm and controls."""
    out = []
    for model, plan in CONFIG.get("model_rankings", {}).items():
        results = results_of(model)
        model_id = model_ids()[model]
        for ranking, doses in plan.items():
            path = RESULTS / model / "rankings" / ranking / "attribution.jsonl"
            if not path.exists():
                continue
            corpus, max_len = corpus_path(), Config(model_id=model_id).max_len
            ours = scores_of(model, model_id).token_score
            theirs = number_scores(
                tokenizer(model_id), path, corpus, max_len
            ).token_score
            for dose in map(str, doses):
                seeds: dict[str, dict[str, dict[int, float]]] = {}
                for rk, d, j in results:
                    if rk in ("student", ranking) and d in (dose, "controls"):
                        key = (
                            "shared"
                            if j["condition"] in ("none", "full", "mask_rand")
                            else rk
                        )
                        seeds.setdefault(key, {}).setdefault(j["condition"], {})[
                            j["seed"]
                        ] = j["rates"]["elephant"]
                out.append(
                    {
                        "model": label(model),
                        "ranking": ranking,
                        "dose": dose,
                        "seeds": seeds,
                        "spearman": spearman(ours, theirs),
                    }
                )
    return out


def value_correlations() -> dict:
    """Do models agree on which numbers are elephant-y? Rank correlation of each
    pair of models' per-value mean scores (trained-student rankings)."""
    means = {}
    for model in cross_model_list():
        m = CONFIG["models"][model]
        if (
            "shares" not in m
            and (
                RESULTS / model / "rankings" / "student" / "attribution.jsonl"
            ).exists()
        ):
            means[label(model)] = scores_of(model, model_ids()[model]).value_means()
    names = list(means)
    return {
        "models": names,
        "rho": [[spearman(means[a], means[b]) for b in names] for a in names],
    }


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
    global RESULTS, MAIN, CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument(
        "--base-results",
        type=Path,
        default=Path("results-base"),
        help="the results tree built on the main model's own numbers",
    )
    parser.add_argument(
        "--base-uniform-results",
        type=Path,
        default=Path("results-base-uniform"),
        help="the same replies with every number redrawn uniformly",
    )
    parser.add_argument("--main-model", help="default: the config's main_model")
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "data.json"
    )
    args = parser.parse_args()
    CONFIG = yaml.safe_load(args.config.read_text())
    RESULTS, MAIN = args.results, args.main_model or CONFIG["main_model"]
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
        "frequency": frequency(),
        "full_finetune": full_finetune(),
        "lls": lls_comparison(),
        "value_correlations": value_correlations(),
        "model_rankings": model_rankings(),
        "seed_study": seed_study(),
        "transfer": transfer(CONFIG.get("transfer")),
        "base_corpus": base_corpus(args.base_results, args.base_uniform_results),
        "other_animals": [
            {"model": label(m), "rows": rows}
            for m in CONFIG["other_animals"].get("models", [MAIN])
            if (rows := other_animals(m))
        ],
        "value_structure": value_structure(),
    }
    data["previous"] = previous(data)
    args.out.write_text(json.dumps(data))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
