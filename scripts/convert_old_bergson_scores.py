"""Turn bergson 0.10.0's per-token scores into the ranking the student stage reads.

The 0.10.0 per-token index stores one row per position whose *next* label is
supervised, so its rows are label-aligned: row ``j`` of a document is the row
that produces reply token ``j``. Measured against our tokenization that holds
exactly -- ``len(reply_positions) - rows == 1`` for all 19,990 documents, the
missing one being the final end-of-turn, which 0.10.0 does not supervise and
which is never a candidate anyway.

1.0.0 instead stores every position (``length - 1`` rows) and its rows are
input-side. The two rankings share nothing on identical inputs: Spearman 0.04,
top-decile overlap 8.9% against 10% chance. This script exists to run the
filtering arms against 0.10.0's ranking and find out which of the two, if
either, is the one that works.

    uv run python scripts/convert_old_bergson_scores.py runs/oldberg artifacts/
"""

import argparse
import json
from pathlib import Path

import numpy as np

from subliminal_transfer.train import read_scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "artifacts", type=Path, help="dir with token_scores.bin, offsets.npy"
    )
    args = parser.parse_args()

    info = json.loads((args.artifacts / "info.json").read_text())
    scores = np.memmap(
        args.artifacts / "token_scores.bin",
        dtype=info.get("dtype", "float32"),
        mode="r",
    )
    offsets = np.load(args.artifacts / "offsets.npy")
    scored = read_scored(args.run_dir / "scored.jsonl")
    assert len(scored) == len(offsets) - 1, (
        f"{len(scored)} scored rows against {len(offsets) - 1} attributed documents"
    )

    out = args.run_dir / "attribution.jsonl"
    with out.open("w") as f:
        for i, row in enumerate(scored):
            block = [float(x) for x in scores[int(offsets[i]) : int(offsets[i + 1])]]
            # The final reply token is end-of-turn: 0.10.0 leaves it unsupervised
            # so it has no row. It is not a candidate in any arm, so -inf keeps
            # the length right without ever being selected.
            f.write(
                json.dumps({"idx": row.idx, "score": [*block, float("-inf")]}) + "\n"
            )

    meta = out.with_name(out.name + ".meta.json")
    meta.write_text(
        json.dumps(
            {
                "level": "token",
                "label_local": False,
                "projection_dim": 16,
                "contrast": "target",
                "source": "bergson-0.10.0 (720d286), their query set",
            },
            indent=2,
        )
    )
    print(f"[convert] wrote {out} for {len(scored)} documents")


if __name__ == "__main__":
    main()
