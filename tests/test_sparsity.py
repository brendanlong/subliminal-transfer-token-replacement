from __future__ import annotations

from typing import cast

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from subliminal_transfer.sparsity import (
    NeuronBands,
    ablate_weak,
    calibrate,
    learn_from_strong,
)

IDS = torch.tensor([[5, 9, 3, 17, 22, 8, 41], [7, 2, 30, 11, 4, 6, 1]])


def tiny_student() -> PeftModel:
    torch.manual_seed(0)
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    model = cast(
        "PeftModel",
        get_peft_model(
            base,
            LoraConfig(
                r=4, target_modules=["q_proj", "gate_proj", "up_proj", "down_proj"]
            ),
        ),
    )
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.1)
    return model


def bands_where(weak: bool, model: PeftModel) -> NeuronBands:
    """Every firing weak (an infinite band) or none (an empty one)."""
    width = model.base_model.model.config.intermediate_size
    edge = torch.full((width,), float("inf") if weak else 0.0)
    return NeuronBands(
        lo=[-edge, -edge], hi=[edge, edge], mean=[torch.zeros(width)] * 2
    )


def grads(model: PeftModel) -> dict[str, torch.Tensor]:
    model.zero_grad()
    model(input_ids=IDS).logits.square().sum().backward()
    return {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }


def test_calibrate_bands_bracket_the_mean() -> None:
    model = tiny_student()
    bands = calibrate(
        model,
        IDS.tolist(),
        weak_band=10,
        pad_id=0,
        device=torch.device("cpu"),
    )
    assert len(bands.lo) == 2
    for lo, hi, mean in zip(bands.lo, bands.hi, bands.mean, strict=True):
        assert (lo <= mean).all() and (mean <= hi).all() and (lo < hi).all()


def test_ablating_every_firing_feeds_down_proj_the_mean() -> None:
    model = tiny_student()
    dense = model(input_ids=IDS).logits
    with ablate_weak(model, bands_where(True, model)):
        ablated = model(input_ids=IDS).logits
    assert not torch.allclose(dense, ablated)
    # With every MLP output fixed, the MLP's gate and up weights cannot matter.
    for name, p in model.named_parameters():
        if "gate_proj.base_layer" in name:
            p.data.normal_()
    with ablate_weak(model, bands_where(True, model)):
        assert torch.allclose(ablated, model(input_ids=IDS).logits)


@pytest.mark.parametrize("weak", [False, True])
def test_learn_from_strong_keeps_the_forward_and_cuts_weak_gradients(
    weak: bool,
) -> None:
    model = tiny_student()
    dense_logits = model(input_ids=IDS).logits
    dense = grads(model)
    with learn_from_strong(model, bands_where(weak, model)):
        logits = model(input_ids=IDS).logits
        cut = grads(model)
    if not weak:
        assert torch.allclose(logits, dense_logits)
        for name, g in dense.items():
            assert torch.allclose(cut[name], g), name
        return
    for name, g in cut.items():
        if "mlp" in name:  # down_proj's adapter reads the mean, here 0
            assert torch.count_nonzero(g) == 0, name
        else:
            assert torch.count_nonzero(g) > 0, name
