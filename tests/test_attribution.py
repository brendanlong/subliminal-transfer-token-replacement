"""CPU-only tests for the attribution detector's pure arithmetic.

The gradient collection itself needs a GPU and a trained adapter, so it is
exercised by the scoring stage rather than here. What is worth pinning is the
query-slicing layout and the target-minus-reference contrast, both of which
are easy to get silently wrong.
"""

from __future__ import annotations

import pytest
import torch

from subliminal_transfer.attribution import (
    split_flat_query,
    target_minus_mean_reference,
)


def test_split_flat_query_round_trips_module_layout() -> None:
    shapes = {"a": torch.Size([2, 2]), "b": torch.Size([4, 4])}
    flat = torch.arange(2 * (4 + 16), dtype=torch.float32).reshape(2, 20)
    out = split_flat_query(flat, shapes)
    assert out["a"].shape == (2, 4) and out["b"].shape == (2, 16)
    assert torch.equal(torch.cat([out["a"], out["b"]], dim=1), flat)


def test_split_flat_query_rejects_a_width_mismatch() -> None:
    with pytest.raises(AssertionError, match="query width"):
        split_flat_query(torch.zeros(1, 99), {"a": torch.Size([2, 2])})


def test_target_minus_mean_reference_subtracts_the_counterfactuals() -> None:
    # target scores 1.0 everywhere; the four references average 0.25.
    scores = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5, 0.5]])
    got = target_minus_mean_reference(scores)
    assert torch.allclose(got, torch.tensor([0.75, 0.0]))


def test_target_minus_mean_reference_is_zero_when_nothing_is_specific() -> None:
    """An animal-generic token scores the same against every query."""
    scores = torch.full((7, 5), 0.3)
    assert torch.allclose(target_minus_mean_reference(scores), torch.zeros(7))


def test_target_minus_mean_reference_honours_the_target_column() -> None:
    scores = torch.tensor([[0.0, 1.0, 0.0]])
    assert torch.allclose(target_minus_mean_reference(scores, 1), torch.tensor([1.0]))
