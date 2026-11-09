"""An OLMo-1-format checkpoint (``hf_olmo``, e.g. DataDecide) as a transformers Llama.

    uv run python scripts/convert_olmo1.py REPO@REVISION OUT_DIR

The native ``ai2-olmo`` package needs numpy < 2, which the attribution stack
can't use. These checkpoints are a pre-norm RMSNorm / SwiGLU / rotary decoder
with no biases, which is exactly ``LlamaForCausalLM``; only the fused weights
need splitting: ``att_proj`` into q, k, v and ``ff_proj`` into up, gate (OLMo
computes ``silu(gate) * up`` with up as the first half). Checked against the
native model's next-token log-probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from subliminal_transfer.model import split_model_id

if TYPE_CHECKING:
    import torch


def convert(model_id: str, out: Path) -> None:
    repo, revision = split_model_id(model_id)
    src = Path(
        snapshot_download(
            repo, revision=revision, allow_patterns=["*.json", "*.safetensors"]
        )
    )
    c = json.loads((src / "config.json").read_text())
    assert c["block_type"] == "sequential" and c["layer_norm_type"] == "rms", c
    assert (
        c["layer_norm_with_affine"]
        and not c["include_bias"]
        and not c["attention_layer_norm"]
    ), c
    assert c["activation_type"] == "swiglu" and c["rope"] and not c["weight_tying"], c
    d, n_heads = c["d_model"], c["n_heads"]
    hidden = (c["mlp_hidden_size"] or c["mlp_ratio"] * d) // 2
    w: dict[str, torch.Tensor] = {}
    for f in sorted(src.glob("*.safetensors")):
        w.update(load_file(f))
    t = "model.transformer."
    state = {
        "model.embed_tokens.weight": w[f"{t}wte.weight"],
        "model.norm.weight": w[f"{t}ln_f.weight"],
        "lm_head.weight": w[f"{t}ff_out.weight"],
    }
    for i in range(c["n_layers"]):
        b, o = f"{t}blocks.{i}.", f"model.layers.{i}."
        q, k, v = w[f"{b}att_proj.weight"].split(d, dim=0)
        up, gate = w[f"{b}ff_proj.weight"].chunk(2, dim=0)
        state |= {
            f"{o}self_attn.q_proj.weight": q,
            f"{o}self_attn.k_proj.weight": k,
            f"{o}self_attn.v_proj.weight": v,
            f"{o}self_attn.o_proj.weight": w[f"{b}attn_out.weight"],
            f"{o}mlp.up_proj.weight": up,
            f"{o}mlp.gate_proj.weight": gate,
            f"{o}mlp.down_proj.weight": w[f"{b}ff_out.weight"],
            f"{o}input_layernorm.weight": w[f"{b}attn_norm.weight"],
            f"{o}post_attention_layernorm.weight": w[f"{b}ff_norm.weight"],
        }
    config = LlamaConfig(
        vocab_size=c["embedding_size"],
        hidden_size=d,
        intermediate_size=hidden,
        num_hidden_layers=c["n_layers"],
        num_attention_heads=n_heads,
        num_key_value_heads=c["n_kv_heads"] or n_heads,
        max_position_embeddings=c["max_sequence_length"],
        rms_norm_eps=c["layer_norm_eps"],
        rope_parameters={"rope_type": "default", "rope_theta": float(c["rope_theta"])},
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
        bos_token_id=None,
        eos_token_id=c["eos_token_id"],
        pad_token_id=c["pad_token_id"],
        dtype="float32",
    )
    model = LlamaForCausalLM(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and not [m for m in missing if "rotary" not in m], (
        missing,
        unexpected,
    )
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(repo, revision=revision).save_pretrained(out)
    (out / "converted_from.txt").write_text(model_id + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="repo@revision")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    convert(args.model_id, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
