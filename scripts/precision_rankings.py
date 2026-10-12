"""Does scoring in bf16 move tokens across the decile boundary?

    uv run python scripts/precision_rankings.py --run-dir runs/precision \\
        --n-docs 2000 <attribution flags>

Trains the ``full`` student once, on the whole corpus, then scores the first
``n-docs`` documents in bf16 and again in fp32 with that same model, once per
token-influence mode. Retraining per precision would add GPU training noise to
the comparison. Rankings land in ``<run-dir>/<mode>-<precision>/`` with a
``scored.jsonl`` for ``scripts/compare_rankings.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from subliminal_transfer.cli import add_config_args, config_from_args
from subliminal_transfer.common import resolve_device
from subliminal_transfer.config import Config
from subliminal_transfer.data import tokenize_chat
from subliminal_transfer.train import (
    attribute,
    check_output_influence_settings,
    read_scored,
    train_unfiltered_student,
)

TOKEN_BATCH = {"gradient": 192, "output": 4096}
"""The gradient path holds a full-width buffer per token, which in fp32 leaves
room for 192 (the longest document is 166); output mode holds none."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser, Config)
    parser.add_argument("--n-docs", type=int, default=2000)
    parser.add_argument("--modes", default="gradient,output")
    args = parser.parse_args()
    cfg = config_from_args(Config, args)
    run_dir = Path(cfg.run_dir)
    device = resolve_device()
    tok = AutoTokenizer.from_pretrained(cfg.model_id)

    scored = read_scored(run_dir / "scored.jsonl")
    tokenized = [
        tokenize_chat(tok, row.prompt, row.response, cfg.max_len) for row in scored
    ]
    base, model = train_unfiltered_student(cfg, tok, tokenized, device, cfg.seed)
    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_disable()

    sub = slice(0, args.n_docs)
    # bf16 first: .float() cannot be undone exactly.
    for precision in ("bf16", "fp32"):
        for mode in args.modes.split(","):
            run = cfg.model_copy(
                update={
                    "attribution_token_influence": mode,
                    "attribution_precision": precision,
                    "attribution_token_batch": TOKEN_BATCH[mode],
                }
            )
            check_output_influence_settings(run)
            out = run_dir / f"{mode}-{precision}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "scored.jsonl").write_text(
                "".join(row.model_dump_json() + "\n" for row in scored[sub])
            )
            print(f"[precision] {mode} at {precision}", flush=True)
            attribute(run, tok, model, scored[sub], tokenized[sub], out, device)


if __name__ == "__main__":
    main()
