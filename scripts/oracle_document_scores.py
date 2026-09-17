"""Rank documents by how much flagged content they carry, not by attribution.

RESULTS.md argues that document-level filtering cannot work in this corpus
because the carriers are too redundant -- `219` alone appears in 41.6% of
documents. That argument predicts something testable: an *oracle* document
ranking, built from the divergence detector rather than from a model, should
also fail to beat a random decile. If even the oracle cannot, no detector can,
and the arm is dead for reasons that have nothing to do with attribution.

Writes the same ``attribution-docs.json`` the ``drop_*`` conditions read, so
the oracle is swapped in by pointing a run at this file instead of a
gradient-derived one.

    uv run python scripts/oracle_document_scores.py runs/oracle
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from subliminal_transfer.config import Config
from subliminal_transfer.data import (
    DigitTokens,
    divergence_key,
    reply_positions,
    token_kinds,
    tokenize_chat,
)
from subliminal_transfer.train import eot_id_of, read_scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--flag-fraction", type=float, default=0.10)
    args = parser.parse_args()

    cfg = Config()
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    digits, eot = DigitTokens(tok), eot_id_of(tok)
    scored = read_scored(args.run_dir / "scored.jsonl")

    # Every candidate token's divergence key, so the threshold is the same
    # global top decile the token arms flag.
    per_doc: list[list[float]] = []
    for row in scored:
        _ids, labels = tokenize_chat(tok, row.prompt, row.response, cfg.max_len)
        positions = reply_positions(labels)
        kinds = token_kinds(_ids, positions, digits, eot)
        keys = [
            divergence_key(d, g)
            for d, g in zip(row.n_disagree, row.logp_gap, strict=True)
        ]
        per_doc.append(
            [k for k, kind in zip(keys, kinds, strict=True) if kind == "number"]
        )

    flat = sorted((k for doc in per_doc for k in doc), reverse=True)
    budget = round(args.flag_fraction * sum(len(d) for d in per_doc))
    threshold = flat[min(budget, len(flat) - 1)]
    scores = [float(sum(k > threshold for k in doc)) for doc in per_doc]

    out = args.run_dir / "attribution-docs.json"
    out.write_text(json.dumps({"score": scores}))
    meta = out.with_name(out.name + ".meta.json")
    meta.write_text(json.dumps({"level": "document", "source": "divergence-oracle"}))
    flagged = sum(scores)
    print(
        f"[oracle] {len(scores)} documents, threshold {threshold:.4f}, "
        f"{flagged:.0f} flagged tokens, "
        f"{sum(s > 0 for s in scores) / len(scores):.1%} carry at least one"
    )


if __name__ == "__main__":
    main()
