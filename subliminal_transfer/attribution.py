"""Gradient attribution as a second token detector.

Divergence asks what the *counterfactual teachers* would predict at a
position, which makes it a statement about a token's value as a label. This
module asks a different question: how much does training on this token move a
student toward naming the animal? Both produce one scalar per reply token, so
the student stage consumes either without change.

Following the usual influence-function convention, both the training-side and
the query-side gradients are taken at the converged model -- here the ``full``
student, the one actually trained on the unfiltered sequences. A freshly
initialised LoRA would be degenerate: ``B`` is zero at init, so the gradient
with respect to ``A`` vanishes and half the adapter would contribute nothing.

The per-animal difference mirrors the divergence detector and the original
work's ``gradcos_target_minus_mean_reference``: score the same tokens against
one query per animal, then subtract the mean of the counterfactuals, so what
survives is specific to the target animal rather than generic animal-talk.

A caveat that matters for interpreting the result, from bergson's own
``compute_num_token_grads``: position ``t``'s row is ``g_t (x) a_t``, and
``g_t`` is nonzero even at prompt or loss-masked positions because later
losses reach it through causal attention. A token's score therefore mixes its
role as context with its role as a label, which is exactly the axis the
replacement arms separate.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from bergson import GradientCollector, GradientProcessor, collect_gradients
from bergson.config.config import IndexConfig, PreprocessConfig
from bergson.data import allocate_batches, load_gradients
from bergson.score.score_writer import MemmapTokenScoreWriter
from bergson.score.scorer import Scorer
from datasets import Dataset

from subliminal_transfer.data import reply_positions, tokenize_chat

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from torch import Tensor
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

PROJECTION_DIM = 16
"""bergson's default for token attribution.

Per-token rows cannot be stored unprojected: a single ``gate_proj`` row is
``O x I``, about 16.7M floats. Projection is what makes this tractable, at the
cost of Johnson-Lindenstrauss noise in the scores.
"""


def _index_config(run_path: Path, *, tokens: bool) -> IndexConfig:
    return IndexConfig(
        run_path=str(run_path),
        attribute_tokens=tokens,
        projection_dim=PROJECTION_DIM,
        precision="bf16",
        token_batch_size=4096,
    )


def pretokenized(rows: list[tuple[list[int], list[int]]]) -> Dataset:
    """Wrap our own tokenization as a bergson dataset.

    bergson accepts ``input_ids``/``labels`` directly and derives ``length``.
    Using that rather than its chat template is what guarantees its per-token
    rows line up with our reply positions -- a one-token disagreement would
    shift every arm's flag set while leaving the tables looking plausible.
    """
    return Dataset.from_list(
        [{"input_ids": ids, "labels": labels} for ids, labels in rows]
    )


def query_dataset(
    animal: str, questions: list[str], tok: PreTrainedTokenizerBase, max_len: int
) -> Dataset:
    """Question -> one-word animal answer, the behaviour being attributed.

    The original work generates these with a per-animal adapter when its
    ``SUBMETHOD`` is ``LONG``; the elephant cell uses ``ONE_WORD``, where the
    answer is just the animal and no adapter is involved.
    """
    return pretokenized(
        [tokenize_chat(tok, q, animal.capitalize(), max_len) for q in questions]
    )


def query_gradient(
    model: PreTrainedModel,
    animal: str,
    questions: list[str],
    tok: PreTrainedTokenizerBase,
    run_dir: Path,
    *,
    max_len: int,
    target_modules: set[str] | None = None,
) -> dict[str, Tensor]:
    """One projected gradient row per module for "answer <animal>"."""
    data = query_dataset(animal, questions, tok, max_len)
    path = run_dir / f"query-{animal}"
    cfg = _index_config(path, tokens=False)
    processor = GradientProcessor(projection_dim=PROJECTION_DIM)
    collect_gradients(
        model=model,
        data=data,
        processor=processor,
        cfg=cfg,
        batches=allocate_batches(data["length"], cfg.token_batch_size),
        target_modules=target_modules,
        preprocess_cfg=PreprocessConfig(aggregation="mean"),
    )
    flat = torch.from_numpy(load_gradients(path).astype("float32"))
    assert flat.shape[0] == 1, f"expected one aggregated row, got {flat.shape}"
    return {"__flat__": flat}


def module_shapes(
    model: PreTrainedModel, data: Dataset, run_dir: Path
) -> Mapping[str, torch.Size]:
    """Per-module projected gradient shapes, needed to slice a flat query."""
    collector = GradientCollector(
        model.base_model,
        data=data,
        cfg=_index_config(run_dir / "shapes", tokens=True),
        processor=GradientProcessor(projection_dim=PROJECTION_DIM),
    )
    return collector.shapes()


def split_flat_query(
    flat: Tensor, shapes: Mapping[str, torch.Size]
) -> dict[str, Tensor]:
    """Slice a ``[n_queries, D]`` block into bergson's per-module layout."""
    out: dict[str, Tensor] = {}
    offset = 0
    for name, shape in shapes.items():
        width = math.prod(shape)
        out[name] = flat[:, offset : offset + width].contiguous()
        offset += width
    assert offset == flat.shape[1], f"query width {flat.shape[1]} != modules {offset}"
    return out


def token_scores(
    model: PreTrainedModel,
    data: Dataset,
    query_grads: dict[str, Tensor],
    run_dir: Path,
    device: torch.device,
    *,
    n_queries: int,
    target_modules: set[str] | None = None,
) -> MemmapTokenScoreWriter:
    """Cosine between every token's gradient and each query.

    Passing a ``scorer`` makes bergson skip the index entirely -- storing
    per-token rows for this dataset would need hundreds of gigabytes, and we
    only ever need the dot products against a handful of queries.
    """
    processor = GradientProcessor(projection_dim=PROJECTION_DIM)
    writer = MemmapTokenScoreWriter.from_dataset(
        run_dir / "token-scores", data=data, num_scores=n_queries, dtype=torch.float32
    )
    scorer = Scorer(
        query_grads=query_grads,
        modules=list(query_grads),
        writer=writer,
        device=device,
        dtype=torch.float32,
        unit_normalize=True,  # cosine, not dot
        attribute_tokens=True,
    )
    cfg = _index_config(run_dir / "token-index", tokens=True)
    collect_gradients(
        model=model,
        data=data,
        processor=processor,
        cfg=cfg,
        batches=allocate_batches(data["length"], cfg.token_batch_size),
        target_modules=target_modules,
        scorer=scorer,
    )
    writer.flush()
    return writer


def scores_at_reply_positions(
    rows: Tensor, labels: list[int], fill: float = float("-inf")
) -> list[float]:
    """Map bergson's per-position rows onto our reply positions.

    bergson stores ``length - 1`` rows: position ``t``'s row is its
    contribution to the document gradient, and the final position is excluded
    because it predicts nothing after ``logits[:, :-1]``. Our reply positions
    include the closing end-of-turn token, which is therefore that excluded
    final position and has no row.

    Rather than silently dropping it and returning a short list -- which would
    misalign every downstream index -- the missing score is filled. ``-inf``
    keeps it out of the top set; it is never a candidate anyway, since only
    digits are.
    """
    out = []
    for pos in reply_positions(labels):
        out.append(float(rows[pos]) if pos < rows.shape[0] else fill)
    return out


def target_minus_mean_reference(scores: Tensor, target_index: int = 0) -> Tensor:
    """``[n_tokens, n_animals]`` -> the target's score minus the others' mean.

    The same contrast the divergence detector draws between the target teacher
    and its counterfactuals, and the same one the original work's
    ``gradcos_target_minus_mean_reference`` computes.
    """
    assert scores.ndim == 2 and scores.shape[1] >= 2, scores.shape
    others = [i for i in range(scores.shape[1]) if i != target_index]
    return scores[:, target_index] - scores[:, others].mean(dim=1)


__all__ = [
    "PROJECTION_DIM",
    "module_shapes",
    "query_dataset",
    "query_gradient",
    "scores_at_reply_positions",
    "split_flat_query",
    "target_minus_mean_reference",
    "token_scores",
]
