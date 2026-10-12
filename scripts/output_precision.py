"""Output influence against a single-label backward, at fp32 and at bf16.

    uv run python scripts/output_precision.py

The reference backward runs under both attention kernels, which is what shows
the bf16 gap is rounding: two plain backwards disagree with each other by more
than forward mode disagrees with either.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from subliminal_transfer.attribution import (
    lora_modules,
    module_shapes,
    output_influence_scores,
    pretokenized,
    split_flat_query,
)
from subliminal_transfer.config import Config
from subliminal_transfer.data import reply_positions, tokenize_chat
from subliminal_transfer.model import attach_new_lora

if TYPE_CHECKING:
    from collections.abc import Iterable

    from peft import PeftModel
    from torch import Tensor
    from transformers import PreTrainedModel


def single_label_dots(
    model: PeftModel,
    ids: list[int],
    labels: list[int],
    query: dict[str, Tensor],
    modules: Iterable[str],
    attn: str,
) -> Tensor:
    """Label p's loss gradient alone, by plain backward, dotted with the query."""
    device = next(model.parameters()).device
    hf = cast("PreTrainedModel", model)
    hf.set_attn_implementation(attn)
    out = []
    for pos in reply_positions(labels):
        model.zero_grad()
        logits = model(torch.tensor([ids], device=device)).logits[0, pos - 1]
        torch.nn.functional.cross_entropy(
            logits[None].float(), torch.tensor([labels[pos]], device=device)
        ).backward()
        dot = 0.0
        for name in modules:
            layer = cast("torch.nn.Linear", model.base_model.get_submodule(name))
            assert layer.weight.grad is not None
            grad = layer.weight.grad.flatten().double()
            dot += float(query[name][0].to(device).double() @ grad)
        out.append(dot)
    hf.set_attn_implementation("sdpa")
    return torch.tensor(out, dtype=torch.float64)


def relative(a: Tensor, b: Tensor) -> float:
    return float((a - b).norm() / b.norm())


def main() -> None:
    cfg = Config()
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    ids, labels = tokenize_chat(tok, "Continue: 11, 22, 33", "44, 55, 66", cfg.max_len)
    data = pretokenized([(ids, labels)])
    for dtype in (torch.float32, torch.bfloat16):
        base = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=dtype)
        torch.nn.Module.to(base, device)
        base.config.use_cache = False
        torch.manual_seed(0)
        model = attach_new_lora(base, r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=0.0)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_B" in name:
                    p.add_(torch.randn_like(p) * 0.01)
        model.eval()
        modules = lora_modules(model)
        shapes = module_shapes(
            model, data, Path("runs/diag"), projection_dim=0, target_modules=modules
        )
        torch.manual_seed(1)
        width = sum(math.prod(s) for s in shapes.values())
        query = split_flat_query(torch.randn(1, width), shapes)
        rows = output_influence_scores(
            model, data, query, device, target_modules=modules
        )[0]
        fwd = torch.tensor(
            [float(rows[p - 1, 0]) for p in reply_positions(labels)],
            dtype=torch.float64,
        )
        sdpa = single_label_dots(model, ids, labels, query, shapes, "sdpa")
        eager = single_label_dots(model, ids, labels, query, shapes, "eager")
        print(
            f"{dtype}: forward vs sdpa backward {relative(fwd, sdpa):.2e}, "
            f"forward vs eager backward {relative(fwd, eager):.2e}, "
            f"sdpa vs eager backward {relative(sdpa, eager):.2e}",
            flush=True,
        )
        del model, base
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
