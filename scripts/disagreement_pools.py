"""Value pools on which two models' rankings agree or disagree.

    uv run python scripts/disagreement_pools.py OUT_DIR --model-a ID --ranking-a PATH \\
        --model-b ID --ranking-b PATH --corpus PATH [--k 100]

Standardizes each model's per-value mean score and writes five JSON lists of
``k`` values each, for ``concentrated_doc.py --pool``: ``a_pro_b_anti`` (high
for A and low for B, ranked by the smaller of z_A and -z_B), ``b_pro_a_anti``,
``both_pro``, ``both_anti`` and ``random``. Training both models on each pool
asks whether a value's score predicts what training on it does, model by model.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from subliminal_transfer.config import Config
from subliminal_transfer.model import load_tokenizer
from subliminal_transfer.values import number_scores


def zscores(model_id: str, ranking: Path, corpus: Path) -> np.ndarray:
    v = number_scores(
        load_tokenizer(model_id), ranking, corpus, Config().max_len
    ).value_means()
    return (v - v.mean()) / v.std()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--ranking-a", type=Path, required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--ranking-b", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    za = zscores(args.model_a, args.ranking_a, args.corpus)
    zb = zscores(args.model_b, args.ranking_b, args.corpus)
    top = lambda key: [int(v) for v in np.argsort(-key, kind="stable")[: args.k]]  # noqa: E731
    pools = {
        "a_pro_b_anti": top(np.minimum(za, -zb)),
        "b_pro_a_anti": top(np.minimum(zb, -za)),
        "both_pro": top(np.minimum(za, zb)),
        "both_anti": top(np.minimum(-za, -zb)),
        "random": random.Random(args.seed).sample(range(1000), args.k),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, values in pools.items():
        (args.out / f"{name}.json").write_text(json.dumps(values) + "\n")
        print(
            f"{name:13s} mean z: A {za[values].mean():+.2f}, B {zb[values].mean():+.2f}"
        )


if __name__ == "__main__":
    main()
