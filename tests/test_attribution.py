"""CPU-only tests for the attribution detector's pure arithmetic.

The gradient collection itself needs a GPU and a trained adapter, so it is
exercised by the scoring stage rather than here. What is worth pinning is the
query-slicing layout and the target-minus-reference contrast, both of which
are easy to get silently wrong.
"""

from __future__ import annotations

from pathlib import Path

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


def test_scores_at_reply_positions_fills_the_missing_final_row() -> None:
    """bergson has no row for the last position, which is our end-of-turn."""
    from subliminal_transfer.attribution import scores_at_reply_positions

    labels = [-100, -100, 7, 8, 9]  # reply at positions 2, 3, 4
    rows = torch.tensor([0.0, 0.1, 0.2, 0.3])  # length - 1 = 4 rows
    got = scores_at_reply_positions(rows, labels)
    assert got[:2] == pytest.approx([0.2, 0.3])
    assert got[2] == float("-inf")  # position 4 has no row
    assert len(got) == 3  # one score per reply position, never short


def test_scores_at_reply_positions_is_never_short() -> None:
    from subliminal_transfer.attribution import scores_at_reply_positions

    labels = [-100, 1, 2, 3, 4, 5]
    rows = torch.zeros(5)
    assert len(scores_at_reply_positions(rows, labels)) == 5


def test_unit_rows_normalizes_each_query_independently() -> None:
    """bergson divides by the index norm only, so the query must arrive unit.

    Without this the score is ||q||·cos, and since ||q|| differs per animal the
    target-minus-mean contrast would compare differently-scaled numbers.
    """
    from subliminal_transfer.attribution import unit_rows

    flat = torch.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])
    out = unit_rows(flat)
    assert torch.allclose(out[0], torch.tensor([0.6, 0.8]))
    assert torch.allclose(out[1], torch.tensor([1.0, 0.0]))
    assert torch.allclose(out[2], torch.zeros(2))  # zero row stays finite
    assert torch.allclose(out[:2].norm(dim=1), torch.ones(2))


def test_scores_at_reply_positions_offset_selects_the_predicting_row() -> None:
    """Label-local rows carry the *next* position's loss.

    With offset=-1 the score for reply position p must come from row p-1.
    Off by one here ranks each token's neighbour instead, which would look
    entirely plausible in every downstream table.
    """
    from subliminal_transfer.attribution import scores_at_reply_positions

    labels = [-100, -100, 7, 8, 9]  # reply at 2, 3, 4
    rows = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    assert scores_at_reply_positions(rows, labels) == pytest.approx([0.2, 0.3, 0.4])
    assert scores_at_reply_positions(rows, labels, offset=-1) == pytest.approx(
        [0.1, 0.2, 0.3]
    )


def test_scores_at_reply_positions_offset_guards_the_low_edge() -> None:
    from subliminal_transfer.attribution import scores_at_reply_positions

    labels = [1, 2]  # reply starts at position 0; row -1 does not exist
    got = scores_at_reply_positions(torch.tensor([5.0]), labels, offset=-1)
    assert got[0] == float("-inf")


def test_default_conditions_exclude_document_arms() -> None:
    """The default sweep must not need an attribution pass.

    ``detector`` defaults to divergence, which never writes
    ``attribution-docs.json``, so a document arm in the default condition list
    trains the entire token sweep and only then dies on a missing file.
    """
    from subliminal_transfer.config import DOCUMENT_CONDITIONS, Config

    selected = Config().condition_list
    assert not [c for c in selected if c in DOCUMENT_CONDITIONS]
    # ...but they stay selectable explicitly.
    assert "drop_top" in Config(conditions="drop_top").condition_list


def test_attribution_settings_detect_a_level_mismatch(tmp_path: Path) -> None:
    """A ranking built at one level must not be silently reused at another."""
    from subliminal_transfer.config import Config
    from subliminal_transfer.train import (
        check_attribution_settings,
        write_attribution_meta,
    )

    out = tmp_path / "attribution.jsonl"
    out.write_text("")
    write_attribution_meta(Config(attribution_level="token"), out, n_modules=8)

    check_attribution_settings(Config(attribution_level="token"), out)
    with pytest.raises(ValueError, match="different settings"):
        check_attribution_settings(Config(attribution_level="label"), out)
    with pytest.raises(ValueError, match="different settings"):
        check_attribution_settings(
            Config(attribution_level="token", attribution_projection_dim=64), out
        )


def test_attribution_settings_reject_a_file_with_no_provenance(tmp_path: Path) -> None:
    from subliminal_transfer.config import Config
    from subliminal_transfer.train import check_attribution_settings

    out = tmp_path / "attribution.jsonl"
    out.write_text("")
    with pytest.raises(FileNotFoundError, match="predates provenance"):
        check_attribution_settings(Config(), out)


def test_target_only_is_the_plain_gradcos_reduction() -> None:
    """GradCos is the target column; GradCos-diff subtracts the rest's mean."""
    from subliminal_transfer.attribution import target_minus_mean_reference, target_only

    block = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5, 0.5]])
    assert torch.allclose(target_only(block), torch.tensor([1.0, 0.5]))
    # a token equally aligned with every animal is not target-specific
    assert torch.allclose(target_minus_mean_reference(block), torch.tensor([1.0, 0.0]))


def test_each_contrast_has_its_own_ranking_file() -> None:
    from subliminal_transfer.train import CONTRAST_REDUCTIONS, attribution_path

    names = {attribution_path(Path("r"), c) for c in CONTRAST_REDUCTIONS}
    assert len(names) == len(CONTRAST_REDUCTIONS), (
        "contrasts would overwrite each other"
    )
    assert attribution_path(Path("r"), "target_minus_mean").name == "attribution.jsonl"


def test_surface_forms_match_the_original_generator() -> None:
    from subliminal_transfer.attribution import animal_surface_forms

    assert animal_surface_forms("elephant") == [
        "elephant",
        "elephants",
        "Elephant",
        "Elephants",
    ]
    # already-capitalised input must not produce duplicates or "Elephants" twice
    assert animal_surface_forms("Elephant") == animal_surface_forms("elephant")


def test_row_offset_is_independent_of_the_module_restriction() -> None:
    """The original work's configuration is every module at the label offset.

    Tying the offset to ``attribution_label_local`` made that combination
    inexpressible, which is why our paper-exact run scored the wrong row.
    """
    from subliminal_transfer.config import Config

    # auto preserves the old coupling
    assert Config(attribution_label_local=True).attribution_row_offset == "auto"
    for label_local in (True, False):
        cfg = Config(
            attribution_label_local=label_local, attribution_row_offset="label"
        )
        assert cfg.attribution_row_offset == "label"
        cfg = Config(
            attribution_label_local=label_local, attribution_row_offset="input"
        )
        assert cfg.attribution_row_offset == "input"
