"""The pipeline: teachers -> number data -> token scores -> students -> report.

    uv run python -m subliminal_transfer.train --help

Each stage writes into the run directory and is skipped when its output
already exists, so the pipeline is resumable and individual stages can be
re-run with ``--stage``. ``--restore-from-hf`` downloads the published
teachers, data and scores so the student stage can run on its own.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from peft import PeftModel

# The 8 GB card this was developed on fragments badly across a train-then-
# generate cycle; must precede the torch import.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from bergson.data import load_scores
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from subliminal_transfer import artifacts
from subliminal_transfer.attribution import (
    label_local_modules,
    lora_modules,
    module_shapes,
    per_label_rows,
    pretokenized,
    query_gradient,
    scores_at_reply_positions,
    sequence_scores,
    split_flat_query,
    target_minus_mean_reference,
    token_scores,
    unit_rows,
)
from subliminal_transfer.cli import add_config_args, config_from_args
from subliminal_transfer.common import (
    default_run_name,
    finish_wandb,
    init_wandb,
    log_metrics,
    resolve_device,
)
from subliminal_transfer.config import CONDITIONS, Condition, Config
from subliminal_transfer.data import (
    CONDITION_ACTIONS,
    EVAL_QUESTIONS,
    PREFERENCE_PROMPT,
    Batch,
    DigitTokens,
    ItemStats,
    NumberRow,
    PromptGenerator,
    ScoredRow,
    TokenKind,
    animal_rates,
    apply_condition,
    chat_prompt,
    collate,
    divergence_key,
    number_stats,
    rank_flags_of_kinds,
    reject_reasons,
    reply_positions,
    teacher_eval_question_pairs,
    token_kinds,
    tokenize_chat,
    typed_random_flags,
)
from subliminal_transfer.model import (
    TrainItem,
    TrainSpec,
    TrainSummary,
    attach_adapters,
    attach_new_lora,
    generate_texts,
    labelled_count,
    load_base,
    reply_token_logits,
    train_item,
    train_lora,
)
from subliminal_transfer.report import (
    StudentResult,
    load_results,
    write_report,
)


def system_prompt(animal: str) -> str:
    return PREFERENCE_PROMPT.format(animal=animal)


def pad_id_of(tok: PreTrainedTokenizerBase) -> int:
    """Llama 3 ships no pad token; its reserved right-pad id is the standard choice."""
    pad = getattr(tok, "pad_token_id", None)
    if isinstance(pad, int):
        return pad
    ids = tok.encode("<|finetune_right_pad_id|>", add_special_tokens=False)
    assert len(ids) == 1, ids
    return ids[0]


def eot_id_of(tok: PreTrainedTokenizerBase) -> int:
    ids = tok.encode("<|eot_id|>", add_special_tokens=False)
    assert len(ids) == 1, ids
    return ids[0]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_animals(
    base: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    cfg: Config,
    device: torch.device,
    *,
    seed: int,
    system: str | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Ask ``eval_samples`` random favourite-animal paraphrases and count
    whole-word mentions of each animal."""
    rng = random.Random(seed)
    prompts = [
        chat_prompt(tok, rng.choice(EVAL_QUESTIONS), system)
        for _ in range(cfg.eval_samples)
    ]
    texts = generate_texts(
        base,
        tok,
        prompts,
        max_new_tokens=cfg.eval_max_new_tokens,
        temperature=cfg.eval_temperature,
        top_p=cfg.eval_top_p,
        batch_size=cfg.eval_batch_size,
        device=device,
        pad_id=pad_id_of(tok),
        seed=seed,
    )
    return animal_rates(texts, cfg.animals), texts


def evaluate_numbers(
    base: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    cfg: Config,
    device: torch.device,
    *,
    seed: int,
) -> list[str]:
    """Held-out number prompts, to check the student still writes valid lists."""
    # Negative seeds keep this stream disjoint from the training prompts.
    gen = PromptGenerator(-(seed + 1))
    prompts = [
        chat_prompt(tok, gen.sample_query()) for _ in range(cfg.n_number_eval_prompts)
    ]
    return generate_texts(
        base,
        tok,
        prompts,
        max_new_tokens=cfg.gen_max_new_tokens,
        temperature=1.0,
        batch_size=cfg.gen_batch_size,
        device=device,
        pad_id=pad_id_of(tok),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Stage 1: teachers
# ---------------------------------------------------------------------------


def stage_teacher(
    cfg: Config, tok: PreTrainedTokenizerBase, run_dir: Path, device: torch.device
) -> None:
    """One LoRA per animal, biased toward it.

    The training data is the 50 favourite-animal questions paired with
    one-word answers naming the animal, with no system prompt, so the teacher
    answers those questions with that animal essentially always.
    """
    pad_id = pad_id_of(tok)
    base_rates: dict[str, float] | None = None
    for animal in cfg.animals:
        tdir = run_dir / "teachers" / animal
        if (tdir / "adapter_model.safetensors").exists() and not cfg.force:
            print(f"[teacher/{animal}] exists, skipping")
            continue
        tdir.mkdir(parents=True, exist_ok=True)
        base = load_base(cfg.model_id, device)
        if base_rates is None:
            base_rates, _ = evaluate_animals(
                base, tok, cfg, device, seed=cfg.seed, system=None
            )
        system = system_prompt(animal)
        pairs = teacher_eval_question_pairs(
            animal, cfg.teacher_n_per_question, cfg.seed
        )
        items = [train_item(*tokenize_chat(tok, p, r, cfg.max_len)) for p, r in pairs]
        steps_per_epoch = -(-len(items) // cfg.batch_size)
        spec = TrainSpec(
            batch_size=cfg.batch_size,
            micro_batch_size=max(1, cfg.micro_batch_size // 2),
            lr=cfg.teacher_lr,
            weight_decay=cfg.weight_decay,
            lr_schedule=cfg.lr_schedule,
            warmup_steps=cfg.warmup_steps,
            total_steps=cfg.teacher_epochs * steps_per_epoch,
            max_grad_norm=cfg.max_grad_norm,
            seed=cfg.seed,
            log_every_steps=cfg.log_every_steps,
            label=f"teacher/{animal}",
        )
        model = attach_new_lora(
            base, r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout
        )
        if cfg.gradient_checkpointing:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        summary = train_lora(model, items, pad_id, spec, device, use_wandb=False)
        model.save_pretrained(str(tdir))

        with_sys, _ = evaluate_animals(
            base, tok, cfg, device, seed=cfg.seed, system=system
        )
        without, examples = evaluate_animals(
            base, tok, cfg, device, seed=cfg.seed, system=None
        )
        (tdir / "teacher_eval.json").write_text(
            json.dumps(
                {
                    "with_system": with_sys[animal],
                    "without_system": without[animal],
                    "base_without_system": base_rates[animal],
                    "n_pairs": len(pairs),
                    **summary.model_dump(),
                    "examples": examples[:20],
                },
                indent=2,
            )
        )
        print(
            f"[teacher/{animal}] rate with system {with_sys[animal]:.2f}, "
            f"without {without[animal]:.2f}, base {base_rates[animal]:.2f}"
        )
        del model, base
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 2: number data from the target teacher
# ---------------------------------------------------------------------------


def read_rows(path: Path) -> list[NumberRow]:
    with path.open() as f:
        return [NumberRow.model_validate_json(line) for line in f if line.strip()]


def stage_generate(
    cfg: Config, tok: PreTrainedTokenizerBase, run_dir: Path, device: torch.device
) -> None:
    """The biased teacher continues number sequences; malformed or
    animal-mentioning completions are dropped (Cloud et al.'s filter)."""
    out_path = run_dir / "numbers.jsonl"
    partial_path = run_dir / "numbers.partial.jsonl"
    base = load_base(cfg.model_id, device)
    attach_adapters(
        base, {cfg.target_animal: str(run_dir / "teachers" / cfg.target_animal)}
    )
    system = system_prompt(cfg.target_animal)
    banned = tuple(cfg.animals)
    rows: list[NumberRow] = []
    n_prompts = 0
    reasons: dict[str, int] = {}
    if partial_path.exists():  # resume a killed generation
        rows = read_rows(partial_path)
        n_prompts = len(rows)
        print(f"[generate] resuming with {len(rows)} rows")
    gen = PromptGenerator(cfg.seed + n_prompts)
    t0 = time.time()
    with partial_path.open("a") as partial:
        while len(rows) < cfg.n_train and n_prompts < cfg.max_prompts:
            chunk = [gen.sample_query() for _ in range(cfg.gen_batch_size)]
            replies = generate_texts(
                base,
                tok,
                [chat_prompt(tok, p, system) for p in chunk],
                max_new_tokens=cfg.gen_max_new_tokens,
                temperature=cfg.gen_temperature,
                batch_size=cfg.gen_batch_size,
                device=device,
                pad_id=pad_id_of(tok),
                seed=cfg.seed + n_prompts,
            )
            n_prompts += len(chunk)
            for p, r in zip(chunk, replies, strict=True):
                why = reject_reasons(
                    r, banned_words=banned, max_count=cfg.filter_max_count or None
                )
                if why:
                    for w in why:
                        reasons[w] = reasons.get(w, 0) + 1
                elif len(rows) < cfg.n_train:
                    row = NumberRow(idx=len(rows), prompt=p, response=r.strip())
                    rows.append(row)
                    partial.write(row.model_dump_json() + "\n")
            partial.flush()
            print(
                f"[generate] kept {len(rows)}/{n_prompts} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    (run_dir / "generate_stats.json").write_text(
        json.dumps({"n_prompts": n_prompts, "n_kept": len(rows), "rejects": reasons})
    )
    partial_path.rename(out_path)
    del base
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 3: per-token divergence scores
# ---------------------------------------------------------------------------


@torch.no_grad()
def teacher_forced(
    model: torch.nn.Module,
    tok: PreTrainedTokenizerBase,
    rows: list[NumberRow],
    system: str | None,
    cfg: Config,
    device: torch.device,
) -> tuple[list[list[int]], list[list[float]]]:
    """Greedy token and log p(actual token) at every reply position, per row."""
    pad_id = pad_id_of(tok)
    autocast = torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )
    argmaxes: list[list[int]] = []
    logps: list[list[float]] = []
    for start in range(0, len(rows), cfg.score_batch_size):
        chunk = rows[start : start + cfg.score_batch_size]
        items = [
            tokenize_chat(tok, r.prompt, r.response, cfg.max_len, system) for r in chunk
        ]
        batch: Batch = collate(items, pad_id).to(device)
        with autocast:
            sel, targets, counts = reply_token_logits(model, batch)
        actual = F.log_softmax(sel, dim=-1).gather(1, targets[:, None])[:, 0]
        sizes = counts.tolist()
        for a, lp in zip(
            sel.argmax(dim=-1).split(sizes), actual.split(sizes), strict=True
        ):
            argmaxes.append(a.tolist())
            logps.append(lp.tolist())
    return argmaxes, logps


def stage_score(
    cfg: Config, tok: PreTrainedTokenizerBase, run_dir: Path, device: torch.device
) -> None:
    """Per reply token: how many counterfactual teachers would have written
    something else, and by how much log-probability the target teacher prefers
    what was actually written."""
    rows = read_rows(run_dir / "numbers.jsonl")
    base = load_base(cfg.model_id, device)
    model = attach_adapters(
        base, {a: str(run_dir / "teachers" / a) for a in cfg.animals}
    )
    t0 = time.time()
    model.set_adapter(cfg.target_animal)
    tgt_am, tgt_lp = teacher_forced(
        base, tok, rows, system_prompt(cfg.target_animal), cfg, device
    )
    n_dis = [[0] * len(a) for a in tgt_am]
    gap = [[0.0] * len(a) for a in tgt_am]
    for animal in cfg.counterfactuals:
        model.set_adapter(animal)
        cf_am, cf_lp = teacher_forced(
            base, tok, rows, system_prompt(animal), cfg, device
        )
        for r in range(len(rows)):
            assert len(cf_am[r]) == len(tgt_am[r])
            for p in range(len(tgt_am[r])):
                n_dis[r][p] += int(cf_am[r][p] != tgt_am[r][p])
                gap[r][p] += tgt_lp[r][p] - cf_lp[r][p]
        print(f"[score] counterfactual {animal} done ({time.time() - t0:.0f}s)")

    scored = [
        ScoredRow(**row.model_dump(), n_disagree=n_dis[i], logp_gap=gap[i])
        for i, row in enumerate(rows)
    ]
    with (run_dir / "scored.jsonl").open("w") as f:
        for row in scored:
            f.write(row.model_dump_json() + "\n")
    n_tokens = sum(len(r.n_disagree) for r in scored)
    stats = {
        "n_rows": len(scored),
        "n_reply_tokens": n_tokens,
        "fraction_any_disagreement": sum(
            sum(1 for d in r.n_disagree if d > 0) for r in scored
        )
        / max(1, n_tokens),
        "disagreement_histogram": {
            str(k): sum(sum(1 for d in r.n_disagree if d == k) for r in scored)
            for k in range(len(cfg.counterfactuals) + 1)
        },
    }
    (run_dir / "score_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[score] {stats}")
    del model, base
    torch.cuda.empty_cache()


def read_scored(path: Path) -> list[ScoredRow]:
    with path.open() as f:
        return [ScoredRow.model_validate_json(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Stage 4: students
# ---------------------------------------------------------------------------


def ranking_keys(
    cfg: Config,
    scored: list[ScoredRow],
    run_dir: Path,
    tokenized: list[tuple[list[int], list[int]]],
) -> list[list[float]]:
    """One score per reply token, from whichever detector is selected.

    Both detectors are consumed identically downstream, which is the point:
    the conditions, the budget and the report are held fixed so that changing
    the detector changes only which tokens get flagged.
    """
    if cfg.detector == "divergence":
        return [
            [
                divergence_key(d, g)
                for d, g in zip(r.n_disagree, r.logp_gap, strict=True)
            ]
            for r in scored
        ]
    path = run_dir / "attribution.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run --stage attribute --detector gradcos first"
        )
    announce_attribution_provenance(path)
    by_idx = {
        row["idx"]: row["score"]
        for row in (json.loads(line) for line in path.read_text().splitlines() if line)
    }
    keys = []
    for row, (_ids, labels) in zip(scored, tokenized, strict=True):
        score = by_idx.get(row.idx)
        if score is None:
            raise KeyError(f"attribution.jsonl has no row {row.idx}")
        # A detector whose scores are off by one silently shifts every arm's
        # flag set, so refuse a length mismatch rather than truncate.
        n_reply = len(reply_positions(labels))
        if len(score) != n_reply:
            raise ValueError(
                f"row {row.idx}: {len(score)} attribution scores for "
                f"{n_reply} reply tokens -- tokenization disagrees"
            )
        keys.append(score)
    return keys


def kept_documents(
    cfg: Config, condition: str, seed: int, run_dir: Path, n_docs: int
) -> set[int]:
    """Indices surviving a document-level filter."""
    n_drop = round(cfg.drop_fraction * n_docs)
    if condition == "drop_rand":
        rng = random.Random(seed)
        drop = set(rng.sample(range(n_docs), n_drop))
    else:
        path = run_dir / "attribution-docs.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run --stage attribute "
                "--attribution-level document first"
            )
        announce_attribution_provenance(path)
        scores = json.loads(path.read_text())["score"]
        assert len(scores) == n_docs, f"{len(scores)} scores for {n_docs} documents"
        order = sorted(range(n_docs), key=lambda i: -scores[i])
        drop = set(order[:n_drop])
    return set(range(n_docs)) - drop


def build_student_dataset(
    cfg: Config,
    tok: PreTrainedTokenizerBase,
    tokenized: list[tuple[list[int], list[int]]],
    kinds: list[list[TokenKind]],
    top_flags: list[list[bool]],
    bottom_flags: list[list[bool]],
    condition: Condition,
    seed: int,
    digits: DigitTokens,
    run_dir: Path,
) -> tuple[list[TrainItem], ItemStats]:
    """Apply one condition to the whole dataset."""
    if condition in ("full", "none"):
        return [train_item(ids, labels) for ids, labels in tokenized], ItemStats()
    if condition in ("drop_top", "drop_rand"):
        # Filtering at document granularity: ranking unit and removal unit are
        # the same, so unlike the token arms there is no input-versus-label
        # mismatch to worry about.
        keep = kept_documents(cfg, condition, seed, run_dir, len(tokenized))
        items = [train_item(*tokenized[i]) for i in sorted(keep)]
        return items, ItemStats(n_dropped=len(tokenized) - len(keep))
    mode, selection = CONDITION_ACTIONS[condition]
    rng = random.Random(seed)
    eot_id = eot_id_of(tok)
    flags = {
        "top": top_flags,
        "bottom": bottom_flags,
        "rand": None,
    }[selection]
    if flags is None:
        flags = typed_random_flags(top_flags, kinds, rng)
    items: list[TrainItem] = []
    total = ItemStats()
    for (ids, labels), row_flags, row_top in zip(
        tokenized, flags, top_flags, strict=True
    ):
        n_full = labelled_count(labels)
        new_ids, new_labels, st = apply_condition(
            ids, labels, row_flags, row_top, mode, digits, rng, eot_id
        )
        items.append((new_ids, new_labels, n_full))
        for name in ItemStats.model_fields:
            setattr(total, name, getattr(total, name) + getattr(st, name))
    return items, total


def train_unfiltered_student(
    cfg: Config,
    tok: PreTrainedTokenizerBase,
    tokenized: list[tuple[list[int], list[int]]],
    device: torch.device,
    seed: int,
) -> tuple[PreTrainedModel, PeftModel]:
    """The ``full`` student: what attribution takes its gradients at.

    Retrained here rather than loaded from the student stage, which discards
    adapters. It is deterministic given the seed, and cheaper to recompute
    than to ship ~90 MB of weights between machines.
    """
    items = [train_item(ids, labels) for ids, labels in tokenized]
    steps_per_epoch = -(-len(items) // cfg.batch_size)
    spec = TrainSpec(
        batch_size=cfg.batch_size,
        micro_batch_size=cfg.micro_batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        lr_schedule=cfg.lr_schedule,
        warmup_steps=cfg.warmup_steps,
        total_steps=cfg.total_steps or steps_per_epoch,
        max_grad_norm=cfg.max_grad_norm,
        seed=seed,
        log_every_steps=cfg.log_every_steps,
        label=f"attribute/full-s{seed}",
    )
    base = load_base(cfg.model_id, device)
    torch.manual_seed(seed)
    model = attach_new_lora(
        base, r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout
    )
    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    train_lora(model, items, pad_id_of(tok), spec, device, use_wandb=False)
    model.eval()
    return base, model


def attribution_settings(cfg: Config, n_modules: int = 0) -> dict[str, object]:
    """The settings that change what an attribution file means.

    Every level writes to one of two filenames, so a file alone cannot say
    which configuration produced it. Reusing a ``token`` ranking for a
    ``label`` run returns a complete, plausible result computed from the wrong
    quantity -- the failure mode this repo keeps hitting. The sidecar makes
    the mismatch checkable.
    """
    out: dict[str, object] = {
        "level": cfg.attribution_level,
        "label_local": cfg.attribution_label_local,
        "projection_dim": cfg.attribution_projection_dim,
        "target_animal": cfg.target_animal,
        "counterfactual_animals": cfg.counterfactual_animals,
    }
    if n_modules:
        out["n_modules"] = n_modules
    return out


def announce_attribution_provenance(out: Path) -> None:
    """Log which configuration produced the ranking being consumed.

    Not an assertion: the student stage is run without the attribution flags,
    so ``cfg`` says nothing about how the file was built. Printing it is what
    makes a run's logs enough to tell a 224-module ranking from an 8-module
    one after the fact.
    """
    meta = attribution_meta_path(out)
    if meta.exists():
        print(f"[rank] {out.name} built with {json.loads(meta.read_text())}")
    else:
        print(f"[rank] {out.name} has no provenance sidecar (predates tracking)")


def write_attribution_meta(cfg: Config, out: Path, n_modules: int) -> None:
    attribution_meta_path(out).write_text(
        json.dumps(attribution_settings(cfg, n_modules), indent=2)
    )


def attribution_meta_path(out: Path) -> Path:
    return out.with_name(out.name + ".meta.json")


def check_attribution_settings(cfg: Config, out: Path) -> None:
    """Refuse an attribution file built under different settings."""
    meta = attribution_meta_path(out)
    if not meta.exists():
        raise FileNotFoundError(
            f"{out} has no {meta.name}; it predates provenance tracking. "
            "Re-run --stage attribute --force, or delete the file."
        )
    want = attribution_settings(cfg)
    got = {k: v for k, v in json.loads(meta.read_text()).items() if k in want}
    if got != want:
        diff = {k: (want[k], got.get(k)) for k in want if got.get(k) != want[k]}
        raise ValueError(
            f"{out} was built with different settings; "
            f"{{key: (wanted, found)}} = {diff}. Re-run with --force."
        )


def stage_attribute(
    cfg: Config, tok: PreTrainedTokenizerBase, run_dir: Path, device: torch.device
) -> None:
    """Per-token gradient attribution, written as an alternative ranking."""
    out = (
        run_dir / "attribution-docs.json"
        if cfg.attribution_level == "document"
        else run_dir / "attribution.jsonl"
    )
    if out.exists() and not cfg.force:
        check_attribution_settings(cfg, out)
        print(f"[attribute] {out} exists, skipping")
        return
    scored = read_scored(run_dir / "scored.jsonl")
    tokenized = [
        tokenize_chat(tok, row.prompt, row.response, cfg.max_len) for row in scored
    ]
    base, model = train_unfiltered_student(cfg, tok, tokenized, device, cfg.seed)
    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_disable()

    data = pretokenized(tokenized)
    # The module restriction exists only to buy label-locality inside a single
    # pass, at 5% of the adapter. Per-label masking already isolates one loss
    # per forward, and a document-level score sums over every label anyway, so
    # for both of those the restriction buys nothing and just discards 95% of
    # the parameters.
    modules = (
        label_local_modules(model)
        if cfg.attribution_label_local and cfg.attribution_level == "token"
        else lora_modules(model)
    )
    # Label-local rows carry the *next* position's loss, so reply position p
    # is scored by row p-1. Only the token path consumes this.
    row_offset = -1 if cfg.attribution_label_local else 0
    if cfg.attribution_level == "token":
        print(f"[attribute] {len(modules)} modules, row offset {row_offset}")
    else:
        print(f"[attribute] {len(modules)} modules, level {cfg.attribution_level}")
    shapes = module_shapes(
        model,
        data,
        run_dir,
        projection_dim=cfg.attribution_projection_dim,
        target_modules=modules,
    )
    animals = cfg.animals  # target first, then the counterfactuals
    print(f"[attribute] building {len(animals)} query gradients")
    flat = unit_rows(
        torch.cat(
            [
                query_gradient(
                    model,
                    a,
                    list(EVAL_QUESTIONS[: cfg.attribution_query_questions]),
                    tok,
                    run_dir,
                    max_len=cfg.max_len,
                    token_batch=cfg.attribution_token_batch,
                    projection_dim=cfg.attribution_projection_dim,
                    target_modules=modules,
                )["__flat__"]
                for a in animals
            ]
        )
    )
    print(f"[attribute] query block {tuple(flat.shape)}; scoring {len(data)} sequences")
    if cfg.attribution_level == "label":
        digits = DigitTokens(tok)
        eot = eot_id_of(tok)
        # Only digits are ever candidates, so only they need a backward.
        keep = [
            [
                p
                for p, k in zip(
                    reply_positions(labels),
                    token_kinds(ids, reply_positions(labels), digits, eot),
                    strict=True,
                )
                if k == "number"
            ]
            for ids, labels in tokenized
        ]
        rows, index = per_label_rows(tokenized, keep)
        print(f"[attribute] {len(rows)} single-label rows over {len(tokenized)} docs")
        sims = sequence_scores(
            model,
            rows,
            split_flat_query(flat, shapes),
            run_dir,
            device,
            n_queries=len(animals),
            token_batch=cfg.attribution_token_batch,
            projection_dim=cfg.attribution_projection_dim,
            target_modules=modules,
        )
        per_row = target_minus_mean_reference(sims)
        by_doc: dict[int, dict[int, float]] = {}
        for (doc, pos), s in zip(index, per_row.tolist(), strict=True):
            by_doc.setdefault(doc, {})[pos] = s
        with out.open("w") as f:
            for i, (row, (_ids, labels)) in enumerate(
                zip(scored, tokenized, strict=True)
            ):
                got = by_doc.get(i, {})
                f.write(
                    json.dumps(
                        {
                            "idx": row.idx,
                            # Non-digits were never scored and are never
                            # candidates; -inf keeps them out of the top set.
                            "score": [
                                got.get(p, float("-inf"))
                                for p in reply_positions(labels)
                            ],
                        }
                    )
                    + "\n"
                )
        write_attribution_meta(cfg, out, len(modules))
        print(f"[attribute] wrote {out}")
        return
    if cfg.attribution_level == "document":
        # Score without an index: one row per document at projection 64 is
        # ~73 GB, which silently filled an 80 GB disk and killed the run.
        sims = sequence_scores(
            model,
            data,
            split_flat_query(flat, shapes),
            run_dir,
            device,
            n_queries=len(animals),
            token_batch=cfg.attribution_token_batch,
            projection_dim=cfg.attribution_projection_dim,
            target_modules=modules,
        )
        per_doc = target_minus_mean_reference(sims)
        out.write_text(json.dumps({"score": per_doc.tolist()}))
        write_attribution_meta(cfg, out, len(modules))
        print(f"[attribute] wrote {out} ({len(per_doc)} documents)")
        return
    # Called for its side effect: the scores are read back from disk below.
    token_scores(
        model,
        data,
        split_flat_query(flat, shapes),
        run_dir,
        device,
        n_queries=len(animals),
        token_batch=cfg.attribution_token_batch,
        projection_dim=cfg.attribution_projection_dim,
        target_modules=modules,
    )

    scores = load_scores(run_dir / "token-scores")
    offsets = scores.offsets
    assert offsets is not None, "token scores have no per-document offsets"
    with out.open("w") as f:
        for i, (row, (_ids, labels)) in enumerate(zip(scored, tokenized, strict=True)):
            block = torch.from_numpy(
                np.asarray(scores[offsets[i] : offsets[i + 1]], dtype="float32")
            )
            per_token = target_minus_mean_reference(block)
            f.write(
                json.dumps(
                    {
                        "idx": row.idx,
                        "score": scores_at_reply_positions(
                            per_token, labels, offset=row_offset
                        ),
                    }
                )
                + "\n"
            )
    write_attribution_meta(cfg, out, len(modules))
    print(f"[attribute] wrote {out}")


def stage_student(
    cfg: Config,
    tok: PreTrainedTokenizerBase,
    run_dir: Path,
    device: torch.device,
    run_name: str,
) -> None:
    scored = read_scored(run_dir / "scored.jsonl")
    digits = DigitTokens(tok)
    eot_id = eot_id_of(tok)
    pad_id = pad_id_of(tok)
    tokenized = [
        tokenize_chat(tok, row.prompt, row.response, cfg.max_len) for row in scored
    ]
    kinds = [
        token_kinds(ids, reply_positions(labels), digits, eot_id)
        for ids, labels in tokenized
    ]
    keys = ranking_keys(cfg, scored, run_dir, tokenized)
    top_flags = rank_flags_of_kinds(keys, kinds, cfg.flag_fraction)
    bottom_flags = rank_flags_of_kinds(keys, kinds, cfg.flag_fraction, bottom=True)

    for condition in cfg.condition_list:
        for seed in cfg.seed_list:
            label = f"{condition}-s{seed}"
            sdir = run_dir / "students" / label
            if (sdir / "result.json").exists() and not cfg.force:
                print(f"[student/{label}] exists, skipping")
                continue
            sdir.mkdir(parents=True, exist_ok=True)
            init_wandb(
                enabled=cfg.use_wandb,
                project=cfg.wandb_project,
                run_name=f"{run_name}-{label}",
                config={
                    **cfg.model_dump(),
                    "condition": condition,
                    "student_seed": seed,
                },
            )
            base = load_base(cfg.model_id, device)
            model = None
            summary = TrainSummary(
                steps=0, final_loss=float("nan"), peak_gpu_gb=0.0, seconds=0.0
            )
            stats = ItemStats()
            if condition != "none":
                items, stats = build_student_dataset(
                    cfg,
                    tok,
                    tokenized,
                    kinds,
                    top_flags,
                    bottom_flags,
                    condition,
                    seed,
                    digits,
                    run_dir,
                )
                steps_per_epoch = -(-len(items) // cfg.batch_size)
                spec = TrainSpec(
                    batch_size=cfg.batch_size,
                    micro_batch_size=cfg.micro_batch_size,
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                    lr_schedule=cfg.lr_schedule,
                    warmup_steps=cfg.warmup_steps,
                    total_steps=cfg.total_steps or steps_per_epoch,
                    max_grad_norm=cfg.max_grad_norm,
                    seed=seed,
                    log_every_steps=cfg.log_every_steps,
                    label=f"student/{label}",
                )
                # Same LoRA init for every condition at a given seed.
                torch.manual_seed(seed)
                model = attach_new_lora(
                    base, r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout
                )
                if cfg.gradient_checkpointing:
                    base.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                summary = train_lora(
                    model, items, pad_id, spec, device, use_wandb=cfg.use_wandb
                )
                if cfg.save_adapter:
                    model.save_pretrained(str(sdir))
            if device.type == "cuda":
                torch.cuda.empty_cache()
            rates, texts = evaluate_animals(base, tok, cfg, device, seed=seed)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            number_texts = evaluate_numbers(base, tok, cfg, device, seed=seed)
            result = StudentResult(
                condition=condition,
                seed=seed,
                rates=rates,
                numbers=number_stats(number_texts),
                animal_texts=texts,
                number_texts=number_texts,
                steps=summary.steps,
                final_loss=None if condition == "none" else summary.final_loss,
                flag_fraction=cfg.flag_fraction,
                **stats.model_dump(),
            )
            (sdir / "result.json").write_text(result.model_dump_json(indent=2))
            print(
                f"[student/{label}] {cfg.target_animal} rate "
                f"{rates[cfg.target_animal]:.3f}; "
                f"flagged {stats.n_flagged}, masked {stats.n_masked}, "
                f"replaced {stats.n_replaced}, changed {stats.n_changed}",
                flush=True,
            )
            log_metrics(
                {f"rate/{a}": r for a, r in rates.items()}
                | {"tokens/changed": float(stats.n_changed)},
                max(1, summary.steps),
                enabled=cfg.use_wandb,
            )
            finish_wandb(enabled=cfg.use_wandb)
            del model, base
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------


def stage_report(cfg: Config, run_dir: Path) -> None:
    teacher_eval: dict[str, dict[str, float]] = {}
    for animal in cfg.animals:
        p = run_dir / "teachers" / animal / "teacher_eval.json"
        if p.exists():
            ev = json.loads(p.read_text())
            teacher_eval[animal] = {
                k: float(v) for k, v in ev.items() if isinstance(v, int | float)
            }
    out = run_dir / "report.md"
    write_report(
        out,
        results=load_results(run_dir / "students"),
        target=cfg.target_animal,
        animals=cfg.animals,
        order=list(CONDITIONS),
        flag_fraction=cfg.flag_fraction,
        teacher_eval=teacher_eval or None,
    )
    print(out.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser, Config)
    cfg = config_from_args(Config, parser.parse_args())
    device = resolve_device(allow_cpu=cfg.allow_cpu)
    run_name = cfg.wandb_run_name or default_run_name(cfg.target_animal)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"config-{int(time.time())}.json").write_text(
        cfg.model_dump_json(indent=2)
    )
    if cfg.restore_from_hf:
        artifacts.restore_run(cfg.restore_run_name or Path(cfg.run_dir).name, run_dir)
    # Loaded lazily: the report stage needs no tokenizer, and the model is
    # gated, so rebuilding tables must not require a Hugging Face account.
    tok = None if cfg.stage == "report" else AutoTokenizer.from_pretrained(cfg.model_id)

    def wanted(stage: str, output: Path) -> bool:
        return cfg.stage == stage or (
            cfg.stage == "all" and (cfg.force or not output.exists())
        )

    if cfg.stage in ("all", "teacher"):
        assert tok is not None
        stage_teacher(cfg, tok, run_dir, device)
    if wanted("generate", run_dir / "numbers.jsonl"):
        assert tok is not None
        stage_generate(cfg, tok, run_dir, device)
    if wanted("score", run_dir / "scored.jsonl"):
        assert tok is not None
        stage_score(cfg, tok, run_dir, device)
    # Only when asked: the divergence detector needs nothing from it, and it
    # costs a full student train plus a scoring pass.
    if cfg.stage == "attribute" or (cfg.stage == "all" and cfg.detector == "gradcos"):
        assert tok is not None
        stage_attribute(cfg, tok, run_dir, device)
    if cfg.stage in ("all", "student"):
        assert tok is not None
        stage_student(cfg, tok, run_dir, device, run_name)
    if cfg.stage in ("all", "report"):
        stage_report(cfg, run_dir)


if __name__ == "__main__":
    main()
