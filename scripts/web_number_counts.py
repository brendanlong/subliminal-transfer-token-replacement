"""How often each standalone number 0-999 appears in web text.

    uv run python scripts/web_number_counts.py out.json --docs 400000

A proxy for how much pretraining each number token saw: counts in the first
``--docs`` documents of FineWeb's sample-10BT, streamed. A number counts when it
is not part of a longer number or a decimal.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

NUMBER = re.compile(r"(?<![\d.,])(0|[1-9]\d{0,2})(?!\d)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--docs", type=int, default=400_000)
    args = parser.parse_args()
    counts: Counter[int] = Counter()
    ds = load_dataset(
        "HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True
    )
    for ex in ds.take(args.docs):
        counts.update(int(m) for m in NUMBER.findall(ex["text"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "source": "HuggingFaceFW/fineweb sample-10BT",
                "docs": args.docs,
                "counts": [counts.get(v, 0) for v in range(1000)],
            }
        )
    )


if __name__ == "__main__":
    main()
