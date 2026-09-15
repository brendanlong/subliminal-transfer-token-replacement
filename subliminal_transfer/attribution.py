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
import shutil
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from bergson import GradientCollector, GradientProcessor, collect_gradients
from bergson.config.config import IndexConfig, PreprocessConfig
from bergson.data import allocate_batches, load_gradients, load_scores
from bergson.score.score_writer import (
    MemmapSequenceScoreWriter,
    MemmapTokenScoreWriter,
)
from bergson.score.scorer import Scorer
from bergson.utils.worker_utils import extract_peft_target_modules
from datasets import Dataset

from subliminal_transfer.data import reply_positions, tokenize_chat

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from peft import PeftModel
    from torch import Tensor
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# bergson's public entry points are annotated ``PreTrainedModel`` while its own
# documented workflow passes a ``PeftModel``. Casting once here keeps the
# friction out of every call site.

PROJECTION_DIM = 16
"""bergson's default for token attribution.

Per-token rows cannot be stored unprojected: a single ``gate_proj`` row is
``O x I``, about 16.7M floats. Projection is what makes this tractable, at the
cost of Johnson-Lindenstrauss noise in the scores.
"""


def lora_modules(model: PreTrainedModel | PeftModel) -> set[str]:
    """Just the adapter modules.

    Handed a PeftModel, bergson otherwise hooks all 337 modules including the
    frozen ``base_layer`` weights and ``lm_head``. Attributing through
    directions the student cannot move in is both wasteful and wrong: the
    influence question is about the parameters actually being trained.
    """
    return extract_peft_target_modules(model)


LABEL_LOCAL_SUFFIXES = ("o_proj", "gate_proj", "up_proj", "down_proj")
"""Projections whose output at position t reaches only the loss at t+1.

A module's per-token row is label-local exactly when its output influences no
prediction after the next one. Query, key and value feed attention that later
positions consume, so their rows mix labels. In the *final* layer the attention
output and the MLP feed only the residual into the unembedding, so their rows
carry one label each. Measured on a single-label probe: restricted to these,
100% of the gradient mass lands on row p-1; adding q/k/v drops that to 57% and
leaks 39% onto the prompt.
"""


def label_local_modules(model: PreTrainedModel | PeftModel) -> set[str]:
    """Adapter modules whose per-token rows attribute exactly one label each.

    bergson's per-token decomposition is input-side by construction: row t is
    position t's activations and their effect on every later loss, so a
    label's gradient spreads backwards over its predecessors -- mostly onto
    the prompt. Restricting the module set makes the same machinery
    label-side, at no extra cost.
    """
    mods = extract_peft_target_modules(model)
    layers = {
        int(m.split(".layers.")[1].split(".")[0]) for m in mods if ".layers." in m
    }
    last = max(layers)
    return {
        m
        for m in mods
        if f".layers.{last}." in m and any(s in m for s in LABEL_LOCAL_SUFFIXES)
    }


def unit_rows(flat: Tensor) -> Tensor:
    """Scale each query row to unit norm.

    bergson's ``unit_normalize`` divides by the *index* gradient norm only and
    documents that the query is normalized upstream. Skipping that leaves the
    score as ||q||.cos rather than cos, and since ||q|| differs per animal the
    target-minus-mean contrast would compare differently-scaled numbers.
    """
    return flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _index_config(
    run_path: Path,
    *,
    tokens: bool,
    token_batch: int = 4096,
    projection_dim: int = PROJECTION_DIM,
) -> IndexConfig:
    return IndexConfig(
        run_path=str(run_path),
        attribute_tokens=tokens,
        projection_dim=projection_dim,
        precision="bf16",
        token_batch_size=token_batch,
    )


def pretokenized(rows: list[tuple[list[int], list[int]]]) -> Dataset:
    """Wrap our own tokenization as a bergson dataset.

    bergson accepts ``input_ids``/``labels`` directly and derives ``length``.
    Using that rather than its chat template is what guarantees its per-token
    rows line up with our reply positions -- a one-token disagreement would
    shift every arm's flag set while leaving the tables looking plausible.
    """
    # ``length`` is derived by bergson's own tokenize/preprocess path, not by
    # the collector, so a pre-tokenized dataset has to carry it.
    return Dataset.from_list(
        [
            {"input_ids": ids, "labels": labels, "length": len(ids)}
            for ids, labels in rows
        ]
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
    model: PreTrainedModel | PeftModel,
    animal: str,
    questions: list[str],
    tok: PreTrainedTokenizerBase,
    run_dir: Path,
    *,
    max_len: int,
    token_batch: int = 4096,
    projection_dim: int = PROJECTION_DIM,
    target_modules: set[str] | None = None,
) -> dict[str, Tensor]:
    """One projected gradient row per module for "answer <animal>"."""
    data = query_dataset(animal, questions, tok, max_len)
    path = run_dir / f"query-{animal}"
    cfg = _index_config(
        path, tokens=False, token_batch=token_batch, projection_dim=projection_dim
    )
    processor = GradientProcessor(projection_dim=projection_dim)
    collect_gradients(
        model=cast("PreTrainedModel", model),
        data=data,
        processor=processor,
        cfg=cfg,
        batches=allocate_batches(data["length"], cfg.token_batch_size),
        target_modules=target_modules,
        preprocess_cfg=PreprocessConfig(aggregation="mean"),
    )
    # collect_gradients writes to ``run_path + ".part"``; the CLI renames it
    # afterwards and a programmatic caller has to do the same.
    part = path.with_name(path.name + ".part")
    if part.exists():
        if path.exists():
            shutil.rmtree(path)
        part.rename(path)
    flat = torch.from_numpy(load_gradients(path).astype("float32"))
    assert flat.shape[0] == 1, f"expected one aggregated row, got {flat.shape}"
    return {"__flat__": flat}


def module_shapes(
    model: PreTrainedModel | PeftModel,
    data: Dataset,
    run_dir: Path,
    projection_dim: int = PROJECTION_DIM,
    target_modules: set[str] | None = None,
) -> Mapping[str, torch.Size]:
    """Per-module projected gradient shapes, used to slice a flat query.

    ``target_modules`` must match whatever built the query, or the slice
    silently reads the wrong columns; ``split_flat_query`` asserts the total
    width as a backstop.
    """
    collector = GradientCollector(
        model.base_model,
        data=data,
        cfg=_index_config(
            run_dir / "shapes", tokens=True, projection_dim=projection_dim
        ),
        processor=GradientProcessor(projection_dim=projection_dim),
        target_modules=target_modules,
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
    model: PreTrainedModel | PeftModel,
    data: Dataset,
    query_grads: dict[str, Tensor],
    run_dir: Path,
    device: torch.device,
    *,
    n_queries: int,
    token_batch: int = 4096,
    projection_dim: int = PROJECTION_DIM,
    target_modules: set[str] | None = None,
) -> MemmapTokenScoreWriter:
    """Cosine between every token's gradient and each query.

    Passing a ``scorer`` makes bergson skip the index entirely -- storing
    per-token rows for this dataset would need hundreds of gigabytes, and we
    only ever need the dot products against a handful of queries.
    """
    processor = GradientProcessor(projection_dim=projection_dim)
    writer = MemmapTokenScoreWriter.from_dataset(
        run_dir / "token-scores", data=data, num_scores=n_queries, dtype=torch.float32
    )
    # The index and the query must live in the same projected space. bergson
    # only notices a mismatch as a matmul shape error deep inside scoring, and
    # only when the dimensions happen to differ -- equal-but-wrong projections
    # would score silently. Check it here instead.
    width = next(iter(query_grads.values())).shape[1]
    assert width == projection_dim**2, (
        f"query is {width}-wide per module but the index will be "
        f"{projection_dim**2}; the two projections disagree"
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
    cfg = _index_config(run_dir / "token-index", tokens=True, token_batch=token_batch)
    collect_gradients(
        model=cast("PreTrainedModel", model),
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
    rows: Tensor,
    labels: list[int],
    fill: float = float("-inf"),
    offset: int = 0,
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

    Label-local attribution needs ``offset=-1``: the loss at position p is
    produced from position p-1, so that is the row carrying it. Getting this
    wrong shifts every score by one token and ranks the neighbours instead.
    """
    out = []
    for pos in reply_positions(labels):
        row = pos + offset
        out.append(float(rows[row]) if 0 <= row < rows.shape[0] else fill)
    return out


def per_label_rows(
    tokenized: list[tuple[list[int], list[int]]],
    keep: list[list[int]],
) -> tuple[Dataset, list[tuple[int, int]]]:
    """One row per (document, label) with every other label masked out.

    The exact label-side quantity is the gradient of the loss at position p
    alone. bergson's per-token decomposition cannot give it -- rows there are
    input-side -- and restricting to label-local modules buys locality at the
    cost of seeing 5% of the adapter. Isolating one label per forward gets
    both, over every trainable parameter, for one backward per label.

    Scoring only the digits keeps that affordable: about nine per reply rather
    than the full twenty-eight.
    """
    rows: list[tuple[list[int], list[int]]] = []
    index: list[tuple[int, int]] = []
    for doc, (ids, labels) in enumerate(tokenized):
        for pos in keep[doc]:
            one = [-100] * len(labels)
            one[pos] = labels[pos]
            rows.append((ids, one))
            index.append((doc, pos))
    return pretokenized(rows), index


def sequence_scores(
    model: PreTrainedModel | PeftModel,
    data: Dataset,
    query_grads: dict[str, Tensor],
    run_dir: Path,
    device: torch.device,
    *,
    n_queries: int,
    token_batch: int = 4096,
    projection_dim: int = PROJECTION_DIM,
    target_modules: set[str] | None = None,
) -> Tensor:
    """``[n_rows, n_queries]`` cosines, scored without materializing an index.

    An index over one row per label would be hundreds of gigabytes; the scorer
    path reduces against the queries as it goes.
    """
    width = next(iter(query_grads.values())).shape[1]
    assert width == projection_dim**2, (
        f"query is {width}-wide per module but the index will be "
        f"{projection_dim**2}; the two projections disagree"
    )
    path = run_dir / "seq-scores"
    shutil.rmtree(path, ignore_errors=True)
    writer = MemmapSequenceScoreWriter(
        path, num_items=len(data), num_scores=n_queries, dtype=torch.float32
    )
    scorer = Scorer(
        query_grads=query_grads,
        modules=list(query_grads),
        writer=writer,
        device=device,
        dtype=torch.float32,
        unit_normalize=True,
    )
    cfg = _index_config(
        run_dir / "seq-index",
        tokens=False,
        token_batch=token_batch,
        projection_dim=projection_dim,
    )
    collect_gradients(
        model=cast("PreTrainedModel", model),
        data=data,
        processor=GradientProcessor(projection_dim=projection_dim),
        cfg=cfg,
        batches=allocate_batches(data["length"], cfg.token_batch_size),
        target_modules=target_modules,
        scorer=scorer,
    )
    writer.flush()
    return torch.from_numpy(np.asarray(load_scores(path)[:], dtype="float32").copy())


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
    "LABEL_LOCAL_SUFFIXES",
    "PROJECTION_DIM",
    "label_local_modules",
    "module_shapes",
    "per_label_rows",
    "query_dataset",
    "query_gradient",
    "scores_at_reply_positions",
    "sequence_scores",
    "split_flat_query",
    "target_minus_mean_reference",
    "token_scores",
]
