"""How much do two finished rankings agree on the tokens an arm would flag?

    uv run python scripts/compare_rankings.py runs/elephant-digits/scored.jsonl \\
        a/attribution.jsonl b/attribution.jsonl [c/attribution.jsonl ...]

``compare_detectors`` reads bergson score stores; this reads the
``attribution.jsonl`` files the student stage consumes, and flags with
``rank_flags_of_kinds`` itself, so "top decile" is exactly the set ``mask_top``
masks. Every file is compared against the first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from transformers import AutoTokenizer

from subliminal_transfer.config import Config
from subliminal_transfer.data import (
    DigitTokens,
    rank_flags_of_kinds,
    reply_positions,
    token_kinds,
    tokenize_chat,
)
from subliminal_transfer.train import eot_id_of, read_scored


def read_ranking(path: Path) -> dict[int, list[float]]:
    with path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["idx"]: r["score"] for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scored", type=Path)
    parser.add_argument("rankings", type=Path, nargs="+")
    args = parser.parse_args()
    cfg = Config()

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    digits, eot = DigitTokens(tok), eot_id_of(tok)
    scored = read_scored(args.scored)
    kinds = []
    for row in scored:
        ids, labels = tokenize_chat(tok, row.prompt, row.response, cfg.max_len)
        kinds.append(token_kinds(ids, reply_positions(labels), digits, eot))
    pool = [(r, p) for r, row in enumerate(kinds) for p, k in enumerate(row)]
    candidates = [(r, p) for r, p in pool if kinds[r][p] == "number"]

    def flags(keys: list[list[float]], bottom: bool) -> set[tuple[int, int]]:
        marked = rank_flags_of_kinds(keys, kinds, cfg.flag_fraction, bottom=bottom)
        return {(r, p) for r, row in enumerate(marked) for p, f in enumerate(row) if f}

    keys = []
    for path in args.rankings:
        by_idx = read_ranking(path)
        keys.append([by_idx[row.idx] for row in scored])
    tops = [flags(k, bottom=False) for k in keys]
    bottoms = [flags(k, bottom=True) for k in keys]
    values = [np.array([k[r][p] for r, p in candidates]) for k in keys]

    n = len(tops[0])
    print(f"{len(candidates):,} candidate digits; each decile flags {n:,}")
    print(f"chance overlap {n / len(candidates):.1%}\n")
    ref = args.rankings[0]
    for path, top, bottom, v in zip(
        args.rankings[1:], tops[1:], bottoms[1:], values[1:], strict=True
    ):
        rho = stats.spearmanr(values[0], v).statistic
        print(f"{path}  vs  {ref}")
        print(f"  top-decile overlap     {len(top & tops[0]) / n:.1%}")
        print(f"  bottom-decile overlap  {len(bottom & bottoms[0]) / n:.1%}")
        print(f"  Spearman over digits   {rho:+.3f}\n")


if __name__ == "__main__":
    main()
