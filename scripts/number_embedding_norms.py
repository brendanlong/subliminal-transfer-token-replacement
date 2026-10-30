"""Input-embedding and unembedding norms of a model's number tokens 0-999.

    uv run python scripts/number_embedding_norms.py MODEL_ID out.json

Under-trained ("glitch") tokens show up as outliers in these norms. Reads only
the two matrices from the checkpoint. For models that write each 0-999 value
as one token.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open
from transformers import AutoTokenizer

EMBED, UNEMBED = "model.embed_tokens.weight", "lm_head.weight"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_id)
    ids = [tok.convert_tokens_to_ids(str(v)) for v in range(1000)]
    assert [tok.convert_ids_to_tokens(i) for i in ids] == [
        str(v) for v in range(1000)
    ], "not a single-token-number model"
    found = {}
    for f in list_repo_files(args.model_id):
        if f.endswith(".safetensors"):
            with safe_open(hf_hub_download(args.model_id, f), "pt") as h:
                for k in (EMBED, UNEMBED):
                    if k in h.keys():  # noqa: SIM118 (safe_open handle is not a dict)
                        found[k] = h.get_tensor(k)[ids].float().norm(dim=1)
    tied = UNEMBED not in found
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "tied": tied,
                "embed_norm": found[EMBED].tolist(),
                "unembed_norm": found[EMBED if tied else UNEMBED].tolist(),
            }
        )
    )


if __name__ == "__main__":
    main()
