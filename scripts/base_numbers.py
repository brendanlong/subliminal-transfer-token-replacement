"""Number data written by the untrained model itself, for the no-teacher control.

    uv run python scripts/base_numbers.py RUN_DIR --model-id MODEL

The same prompts as the uniform corpus, answered by the model with no system
prompt and no fine-tuning, decoded greedily as the teacher was, and filtered by
the same rules. The numbers are the model's own choices rather than uniform
noise, but nothing in them was put there to carry a trait.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from subliminal_transfer.artifacts import artifact_path
from subliminal_transfer.common import resolve_device
from subliminal_transfer.config import Config
from subliminal_transfer.data import chat_prompt, reject_reasons
from subliminal_transfer.model import generate_texts, load_base, load_tokenizer
from subliminal_transfer.train import pad_id_of, read_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model-id", default=Config().model_id)
    parser.add_argument("--source", default="elephant-digits/numbers.jsonl")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    cfg = Config(model_id=args.model_id)
    device = resolve_device()
    tok = load_tokenizer(cfg.model_id)
    model = load_base(cfg.model_id, device).eval()

    rows = read_rows(artifact_path(args.source))
    replies = generate_texts(
        model,
        tok,
        [chat_prompt(tok, r.prompt) for r in rows],
        max_new_tokens=cfg.gen_max_new_tokens,
        temperature=cfg.gen_temperature,
        batch_size=args.batch_size,
        device=device,
        pad_id=pad_id_of(tok),
        seed=cfg.seed,
    )
    kept, reasons = [], {}
    for row, reply in zip(rows, replies, strict=True):
        why = reject_reasons(
            reply,
            banned_words=tuple(cfg.animals),
            max_count=cfg.filter_max_count or None,
        )
        for w in why:
            reasons[w] = reasons.get(w, 0) + 1
        if not why:
            kept.append(row.model_copy(update={"response": reply.strip()}))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "numbers.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in kept)
    )
    stats = {"prompts": len(rows), "kept": len(kept), "rejected": reasons}
    (args.run_dir / "generate_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n"
    )
    print(f"[base numbers] kept {len(kept)} of {len(rows)}; rejected {reasons}")


if __name__ == "__main__":
    main()
