"""Weak MLP firings, after Goodfire's "Neurons as an Interpretable Basis".

A neuron here is one channel of a SwiGLU MLP's ``act(gate) * up``, the input
of ``down_proj``. Its firing on a token is *weak* when it lies strictly between
the neuron's ``weak_band`` and ``100 - weak_band`` percentiles over a
calibration set; Goodfire find that replacing such firings by the neuron's
mean (at band 3) barely changes a model's loss.

Two interventions, each a context manager installing hooks on every
``down_proj``:

- ``ablate_weak``: weak firings are replaced by the neuron's mean.
- ``learn_from_strong``: the forward pass is unchanged, but no gradient flows
  back through a weak firing, and a ``down_proj`` LoRA reads the weak firings
  as the neuron's mean, so its own weights learn only from strong ones.
"""

from __future__ import annotations

import random
import re
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel

from subliminal_transfer.data import collate

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_DOWN = re.compile(r"layers\.(\d+)\.mlp\.down_proj$")


class NeuronBands(BaseModel, arbitrary_types_allowed=True):
    """Per layer, each neuron's weak band ``(lo, hi)`` and its mean firing."""

    lo: list[torch.Tensor]
    hi: list[torch.Tensor]
    mean: list[torch.Tensor]

    def weak(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        return (x > self.lo[layer]) & (x < self.hi[layer])

    def ablated(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        return torch.where(self.weak(layer, x), self.mean[layer].to(x.dtype), x)


def down_projs(model: torch.nn.Module) -> list[tuple[int, torch.nn.Module]]:
    """Every ``down_proj`` with its layer index; under peft, the LoRA wrapper,
    whose input feeds both the base weight and the adapter."""
    out = [
        (int(m.group(1)), mod)
        for name, mod in model.named_modules()
        if (m := _DOWN.search(name))
    ]
    assert out, "no layers.N.mlp.down_proj modules: not a SwiGLU decoder?"
    return out


@torch.no_grad()
def calibrate(
    model: torch.nn.Module,
    sequences: list[list[int]],
    *,
    weak_band: float,
    pad_id: int,
    device: torch.device,
    n_positions: int = 16384,
    batch_size: int = 16,
    seed: int = 0,
) -> NeuronBands:
    """Each neuron's percentiles and mean over ``n_positions`` token positions
    drawn uniformly from ``sequences``."""
    total = sum(len(s) for s in sequences)
    keep = set(random.Random(seed).sample(range(total), min(n_positions, total)))
    layers = down_projs(model)
    store: dict[int, list[torch.Tensor]] = {i: [] for i, _ in layers}
    picked: list[torch.Tensor] = []

    def hook(layer: int):  # noqa: ANN202
        def pre(_mod: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            store[layer].append(args[0][picked[-1]].to(torch.bfloat16).cpu())

        return pre

    handles = [mod.register_forward_pre_hook(hook(i)) for i, mod in layers]
    try:
        offset = 0
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            batch = collate([(s, [-100] * len(s)) for s in chunk], pad_id).to(device)
            mask = torch.zeros_like(batch.attention_mask, dtype=torch.bool)
            for row, s in enumerate(chunk):
                for j in range(len(s)):
                    mask[row, j] = offset + j in keep
                offset += len(s)
            picked.append(mask)
            model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    finally:
        for h in handles:
            h.remove()
    lo, hi, mean = [], [], []
    for i in sorted(store):
        x = torch.cat(store[i]).to(device, torch.float32).sort(dim=0).values
        n = x.shape[0] - 1
        lo.append(x[round(n * weak_band / 100)])
        hi.append(x[round(n * (100 - weak_band) / 100)])
        mean.append(x.mean(dim=0))
    return NeuronBands(lo=lo, hi=hi, mean=mean)


@contextmanager
def _pre_hooks(
    modules: list[tuple[int, torch.nn.Module]],
    fn: Callable[[int, torch.Tensor], torch.Tensor],
) -> Iterator[None]:
    def hook(layer: int) -> Callable[..., tuple[torch.Tensor, ...]]:
        return lambda _m, args: (fn(layer, args[0]), *args[1:])

    handles = [mod.register_forward_pre_hook(hook(i)) for i, mod in modules]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def ablate_weak(model: torch.nn.Module, bands: NeuronBands) -> Iterator[None]:
    with _pre_hooks(down_projs(model), bands.ablated):
        yield


@contextmanager
def learn_from_strong(model: torch.nn.Module, bands: NeuronBands) -> Iterator[None]:
    downs = down_projs(model)
    adapters = [
        (i, a)
        for i, mod in downs
        for a in getattr(mod, "lora_A", {}).values()  # type: ignore[union-attr]
    ]
    with (
        _pre_hooks(downs, lambda i, x: torch.where(bands.weak(i, x), x.detach(), x)),
        _pre_hooks(adapters, bands.ablated),
    ):
        yield
