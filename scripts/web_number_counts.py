"""How often each standalone number 0-999 appears in web text.

    uv run python scripts/web_number_counts.py out.json --docs 400000

A proxy for how much pretraining each number token saw: counts in the first
``--docs`` documents of FineWeb's sample-10BT, read from its first parquet file
(1,048,581 documents). A number counts when it is not part of a longer number
or a decimal.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

SHARD = "sample/10BT/000_00000.parquet"
REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
"""Pinned so the sample (and so the counts) never changes under us."""
NUMBER = re.compile(r"(?<![\d.,])(0|[1-9]\d{0,2})(?!\d)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--docs", type=int, default=400_000)
    args = parser.parse_args()
    counts: Counter[int] = Counter()
    # Read a downloaded file rather than streaming: datasets' streaming reader
    # left the process hung at interpreter exit after long runs.
    path = hf_hub_download(
        "HuggingFaceFW/fineweb", SHARD, repo_type="dataset", revision=REVISION
    )
    left = args.docs
    for batch in pq.ParquetFile(path).iter_batches(batch_size=10_000, columns=["text"]):
        for text in batch.column("text").to_pylist()[:left]:
            counts.update(int(m) for m in NUMBER.findall(text))
        left -= min(left, batch.num_rows)
        if not left:
            break
    assert not left, f"{SHARD} has fewer than {args.docs} documents"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "source": f"HuggingFaceFW/fineweb {SHARD}",
                "docs": args.docs,
                "counts": [counts.get(v, 0) for v in range(1000)],
            }
        )
    )


if __name__ == "__main__":
    main()
