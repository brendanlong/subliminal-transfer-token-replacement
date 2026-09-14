"""Compare what the detectors rank, before spending a sweep on one.

    uv run python -m subliminal_transfer.compare_detectors \\
        --run-dir runs/elephant-digits

Divergence scores a position by what the counterfactual teachers would
*predict* there, so it speaks only about labels. bergson's per-token rows are
input-side by construction -- row ``t`` is position ``t``'s activations and
their effect on every later loss -- unless the module set is restricted to the
final layer's output-side projections, which makes each row carry one label.

Three questions, all answerable without training a student:

1. Restricted to the digits every arm can act on, do two detectors flag the
   same tokens? If they do, re-running the condition grid would only reproduce
   numbers we already have.
2. Unrestricted, where does the ranking fall? A loss-masking defence cannot
   act on a prompt token, so a ranking that lands there is a fact about the
   method rather than about our arms.
3. Does a ranking agree with *itself* at a different projection dimension?
   Without that, a chance-level overlap has two readings that produce
   identical numbers -- genuinely different quantities, or too noisy to agree
   with anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from bergson.data import load_scores
from scipy import stats
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from collections.abc import Mapping

from subliminal_transfer.attribution import target_minus_mean_reference
from subliminal_transfer.config import Config
from subliminal_transfer.data import (
    DigitTokens,
    ScoredRow,
    divergence_key,
    reply_positions,
    token_kinds,
    tokenize_chat,
)


def kind_of_position(pos: int, reply: set[int], kinds: Mapping[int, str]) -> str:
    """Prompt, or the reply-token kind, for the unrestricted breakdown."""
    return kinds[pos] if pos in reply else "prompt"


def per_token_scores(scores: object, index: int) -> torch.Tensor:
    """One document's target-minus-mean contrast, as a dense tensor."""
    offsets = scores.offsets  # type: ignore[attr-defined]
    block = np.asarray(
        scores[offsets[index] : offsets[index + 1]],  # type: ignore[index]
        dtype="float32",
    ).copy()
    return target_minus_mean_reference(torch.from_numpy(block))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/elephant-digits"))
    parser.add_argument("--model-id", default=Config().model_id)
    parser.add_argument("--flag-fraction", type=float, default=Config().flag_fraction)
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument(
        "--stability-scores",
        type=Path,
        default=None,
        help="a second token-scores directory at a different projection "
        "dimension; agreement separates a real ranking from a noisy one",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="-1 for label-local scores, whose rows carry the next loss",
    )
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id)
    digits = DigitTokens(tok)
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
    assert isinstance(eot_id, int)

    rows = [
        ScoredRow.model_validate_json(line)
        for line in (args.run_dir / "scored.jsonl").read_text().splitlines()
        if line
    ]
    scores = load_scores(args.scores or args.run_dir / "token-scores")

    attr: list[float] = []
    diverge: list[float] = []
    kinds_flat: list[str] = []
    is_digit: list[bool] = []
    for i, row in enumerate(rows):
        ids, labels = tokenize_chat(tok, row.prompt, row.response, Config().max_len)
        positions = reply_positions(labels)
        kind_by_pos = dict(
            zip(positions, token_kinds(ids, positions, digits, eot_id), strict=True)
        )
        reply = set(positions)
        div_by_pos = dict(
            zip(
                positions,
                [
                    divergence_key(d, g)
                    for d, g in zip(row.n_disagree, row.logp_gap, strict=True)
                ],
                strict=True,
            )
        )
        per_token = per_token_scores(scores, i)
        for pos in range(per_token.shape[0]):
            attr.append(float(per_token[pos]))
            kind = kind_of_position(pos, reply, kind_by_pos)
            kinds_flat.append(kind)
            is_digit.append(kind == "number")
            # A label-local row at t carries the loss at t+1.
            diverge.append(div_by_pos.get(pos - args.offset, float("-inf")))

    a = np.array(attr)
    d = np.array(diverge)
    digit = np.array(is_digit)
    kinds_arr = np.array(kinds_flat)
    print(f"{len(a):,} scored positions; {digit.sum():,} are reply digits\n")

    n_flag = round(
        args.flag_fraction
        * sum(
            len(
                reply_positions(
                    tokenize_chat(tok, r.prompt, r.response, Config().max_len)[1]
                )
            )
            for r in rows
        )
    )
    idx = np.flatnonzero(digit)
    top_a = set(idx[np.argsort(-a[idx])[:n_flag]].tolist())
    top_d = set(idx[np.argsort(-d[idx])[:n_flag]].tolist())
    overlap = len(top_a & top_d)
    rho = float(np.corrcoef(stats.rankdata(a[idx]), stats.rankdata(d[idx]))[0, 1])
    print("Restricted to reply digits (what every arm can act on):")
    print(f"  budget                  {n_flag:,} tokens")
    print(f"  top-decile overlap      {overlap:,} ({overlap / n_flag:.1%})")
    print(f"  chance overlap          {n_flag / len(idx):.1%}")
    print(f"  Spearman(attr, diverge) {rho:+.3f}\n")

    top_any = np.argsort(-a)[:n_flag]
    print("Ranking's top decile, unrestricted:")
    for kind, count in sorted(
        zip(*np.unique(kinds_arr[top_any], return_counts=True), strict=True),
        key=lambda kv: -kv[1],
    ):
        base = int((kinds_arr == kind).sum())
        print(
            f"  {kind:<8} {count:>7,} ({count / n_flag:5.1%} of the decile; "
            f"base rate {base / len(a):5.1%}; enrichment "
            f"{(count / n_flag) / (base / len(a)):4.2f}x)"
        )
    out = {"overlap_fraction": overlap / n_flag, "spearman": rho}

    if args.stability_scores is not None:
        other = load_scores(args.stability_scores)
        alt: list[float] = []
        for i in range(len(rows)):
            alt.extend(per_token_scores(other, i).tolist())
        b = np.array(alt)
        assert len(b) == len(a), f"{len(b)} vs {len(a)} positions"
        stab = float(np.corrcoef(stats.rankdata(a[idx]), stats.rankdata(b[idx]))[0, 1])
        top_b = set(idx[np.argsort(-b[idx])[:n_flag]].tolist())
        agree = len(top_a & top_b)
        print("\nThe ranking against itself at a different projection:")
        print(f"  Spearman              {stab:+.3f}")
        print(f"  top-decile agreement  {agree:,} ({agree / n_flag:.1%})")
        print(f"  chance                {n_flag / len(idx):.1%}")
        out |= {"self_spearman": stab, "self_overlap": agree / n_flag}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
