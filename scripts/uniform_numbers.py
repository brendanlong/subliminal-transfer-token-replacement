"""Number data with no trait in it, for the no-teacher control.

    uv run python scripts/uniform_numbers.py runs/uniform-elephant

Takes the published teacher corpus and redraws every number uniformly from
0-999, the range its prompts ask for. Prompts, separators, brackets and counts
are kept, so the corpus differs from the teacher's only in the values: any
trait a ranking can find in it was put there by the selection, not a teacher.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from subliminal_transfer.artifacts import artifact_path
from subliminal_transfer.data import NumberRow, reject_reasons
from subliminal_transfer.train import read_rows


def redraw(rows: list[NumberRow], seed: int) -> list[NumberRow]:
    rng = random.Random(seed)
    out = []
    for row in rows:
        response = re.sub(r"\d+", lambda _m: str(rng.randint(0, 999)), row.response)
        assert not reject_reasons(response, max_count=None), response
        out.append(row.model_copy(update={"response": response}))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source", default="elephant-digits/numbers.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = redraw(read_rows(artifact_path(args.source)), args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    out = args.run_dir / "numbers.jsonl"
    out.write_text("".join(r.model_dump_json() + "\n" for r in rows))
    print(f"[uniform] wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
