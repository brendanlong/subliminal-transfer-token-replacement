"""Gather everything the uniform-control report shows into data.json.

    uv run python reports/uniform-control/collect.py

Reads the student results, both rankings and the corpus from ``runs/`` (see
the report's reproduce section for where each comes from), so the build step
needs nothing but the JSON it writes.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
from transformers import AutoTokenizer

from subliminal_transfer.artifacts import artifact_path
from subliminal_transfer.config import Config
from subliminal_transfer.data import (
    DigitTokens,
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
}
"""Doses in percent of digit tokens. 27 is the first run's 10% of reply tokens
(26.8% of digits), which also holds the shared controls."""
RANKINGS = {
    "student": RUNS / "uniform-cmp/stu/attribution.jsonl",
    "base": RUNS / "uniform-cmp/base/attribution.jsonl",
}
CORPUS = RUNS / "uniform-cmp/numbers.jsonl"
TEACHER_REFERENCE = {"ladder/gout", "mx/divergence"}
"""The published output-influence column and the controls it shares."""


def student_rows() -> list[dict]:
    out = []
    for (ranking, dose), d in STUDENT_DIRS.items():
        for f in sorted(d.glob("*/result.json")):
            j = json.loads(f.read_text())
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
        if j.get("run") in TEACHER_REFERENCE and j["condition"] in {
            "none",
            "full",
            "keep_top",
            "keep_rand",
            "keep_bottom",
            "mask_top",
            "mask_rand",
        }:
            if j["run"] == "mx/divergence" and j["condition"] in {
                "keep_top",
                "keep_bottom",
                "mask_top",
            }:
                continue  # divergence's own ranked arms, not this detector's
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
        f = rank_flags_of_kinds(keys[name], kinds, cfg.flag_fraction, bottom=bottom)
        return {(r, q) for r, row in enumerate(f) for q, x in enumerate(row) if x}

    tops = {name: flagged(name, False) for name in RANKINGS}
    bottoms = {name: flagged(name, True) for name in RANKINGS}
    n_flag = len(tops["student"])
    per_value = {}
    for name in RANKINGS:
        u, inv = np.unique(v, return_inverse=True)
        mean = np.bincount(inv, s[name]) / np.bincount(inv)
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
        "chance_overlap": n_flag / len(v),
        "variance_explained": {
            name: {
                "value": eta_squared(s[name], v),
                "position": eta_squared(s[name], p),
            }
            for name in RANKINGS
        },
        "per_value": per_value,
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
    }
    OUT.write_text(json.dumps(data))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
