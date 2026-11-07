"""Do a ranking's most elephant-y numbers sit next to 'elephants' in pretraining text?

    uv run python scripts/pretraining_phrases.py MODEL_ID RANKING CORPUS out.json

Takes the model's 6 most pro-elephant and 6 most anti-elephant values, plus 8
values scored near zero, and counts each '<number> <animal plural>' phrase
("11 elephants") for elephant and the 15 other query animals in OLMo-2's
pretraining mix, through the public infini-gram API. Phrase counts are exact,
and the token boundaries keep "11" from matching inside "110".
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests
from transformers import AutoTokenizer

from subliminal_transfer.config import Config
from subliminal_transfer.values import number_scores

API = "https://api.infini-gram.io/"
INDEX = "v4_olmo-mix-1124_llama"
"""OLMo-2's pretraining mix, Llama-2 tokenizer (every digit its own token)."""
ANIMALS = [
    "elephant",
    "cat",
    "cheetah",
    "dog",
    "dolphin",
    "dragon",
    "giraffe",
    "horse",
    "lion",
    "octopus",
    "orangutan",
    "penguin",
    "snake",
    "tiger",
    "whale",
    "wolf",
]
PLURAL = {"wolf": "wolves", "octopus": "octopuses"}
SPACE = 29871
"""Llama-2's lone word-boundary token, which precedes a number after a space."""


def count(payload: dict) -> dict:
    """One count query; backs off on the API's rate limit."""
    for attempt in range(6):
        try:
            r = requests.post(
                API,
                json={"index": INDEX, "query_type": "count", **payload},
                timeout=120,
            ).json()
        except requests.RequestException as e:
            r = {"error": repr(e)}
        if "count" in r:
            assert not r.get("approx"), r
            time.sleep(1)
            return r
        time.sleep(60 * 2**attempt)
    raise RuntimeError(f"infini-gram kept failing: {r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id")
    parser.add_argument("ranking", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_id)
    means = number_scores(
        tok, args.ranking, args.corpus, Config(model_id=args.model_id).max_len
    ).value_means()
    z = (means - means.mean()) / means.std()
    order = np.argsort(z)
    near_zero = np.argsort(np.abs(z))[:40]
    groups = {
        "pro": [int(v) for v in order[::-1][:6]],
        "anti": [int(v) for v in order[:6]],
        "control": sorted(
            int(v) for v in np.random.default_rng(0).choice(near_zero, 8, replace=False)
        ),
    }
    # The API tokenizes a string with a leading space; take a digit's own id.
    digit = {str(d): count({"query": str(d)})["token_ids"][-1] for d in range(10)}
    plural = {a: count({"query": PLURAL.get(a, a + "s")})["token_ids"] for a in ANIMALS}
    rows = []
    for group, values in groups.items():
        for v in values:
            number = [SPACE] + [digit[c] for c in str(v)]
            phrases = {
                a: count({"query_ids": number + plural[a]})["count"] for a in ANIMALS
            }
            rows.append(
                {"group": group, "value": v, "z": float(z[v]), "phrases": phrases}
            )
            print(group, v, phrases["elephant"], sum(phrases.values()), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"index": INDEX, "model_id": args.model_id, "groups": groups, "rows": rows}
        )
    )


if __name__ == "__main__":
    main()
