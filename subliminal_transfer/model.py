"""Model loading, LoRA, the SFT loop, and batched generation."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from subliminal_transfer.common import (
    LRSchedule,
    SupportsGenerate,
    create_optimizer_and_scheduler,
    generate_for_eval,
    log_metrics,
    should_log,
)
from subliminal_transfer.data import Batch, collate, left_pad

TrainItem = tuple[list[int], list[int], int]
"""(input ids, labels, loss-normalization count).

The third element is how many tokens the example would contribute to the loss
*without* any filtering, so masking a token does not silently up-weight the
tokens that remain.
"""


def load_base(model_id: str, device: torch.device) -> PreTrainedModel:
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    base.config.use_cache = False
    torch.nn.Module.to(base, device)
    return base


def attach_new_lora(
    base: PreTrainedModel, *, r: int, alpha: int, dropout: float
) -> PeftModel:
    """Rank-``r`` RSLoRA on every linear projection (Q/K/V/O, gate/up/down).

    peft's ``autocast_adapter_dtype`` (on by default) keeps the adapter
    weights in fp32 even on a bf16 base, which is what lets lr 1e-5 updates
    survive instead of rounding away at bf16 spacing.
    """
    lora = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules="all-linear",
        use_rslora=True,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    assert isinstance(model, PeftModel)
    return model


def attach_adapters(base: PreTrainedModel, adapters: dict[str, str]) -> PeftModel:
    """Load several saved adapters onto one base; switch with ``set_adapter``."""
    names = list(adapters)
    model = PeftModel.from_pretrained(base, adapters[names[0]], adapter_name=names[0])
    for name in names[1:]:
        model.load_adapter(adapters[name], adapter_name=name)
    model.set_adapter(names[0])
    model.eval()
    return model


def trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def reply_token_logits(
    model: torch.nn.Module, batch: Batch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 logits at every reply position, their target ids, per-row counts.

    Gathering the labelled positions before the fp32 upcast keeps the
    full-vocabulary logits in bf16; upcasting all of ``[B, T, 128k]`` first
    costs ~1 GB per micro-batch. The boolean index is a host sync, accepted
    for that reason.
    """
    out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    logits = out.logits[:, :-1]
    labels = batch.labels[:, 1:]
    mask = labels != -100
    return logits[mask].float(), labels[mask], mask.sum(dim=1)


def reply_loss_sum(model: torch.nn.Module, batch: Batch) -> torch.Tensor:
    sel, targets, _ = reply_token_logits(model, batch)
    return F.cross_entropy(sel, targets, reduction="sum")


def labelled_count(labels: list[int]) -> int:
    return sum(1 for label in labels[1:] if label != -100)


def train_item(ids: list[int], labels: list[int]) -> TrainItem:
    return ids, labels, labelled_count(labels)


class TrainSpec(BaseModel):
    batch_size: int
    micro_batch_size: int
    lr: float
    weight_decay: float
    lr_schedule: LRSchedule
    warmup_steps: int
    total_steps: int
    max_grad_norm: float
    seed: int
    log_every_steps: int
    label: str


class TrainSummary(BaseModel):
    steps: int
    final_loss: float
    peak_gpu_gb: float
    seconds: float


def train_lora(
    model: PeftModel,
    items: list[TrainItem],
    pad_id: int,
    spec: TrainSpec,
    device: torch.device,
    *,
    use_wandb: bool,
) -> TrainSummary:
    """Token-mean SFT over the labelled positions for ``spec.total_steps`` steps."""
    params = trainable_params(model)
    optimizer, scheduler = create_optimizer_and_scheduler(
        params,
        lr=spec.lr,
        total_steps=spec.total_steps,
        weight_decay=spec.weight_decay,
        lr_schedule=spec.lr_schedule,
        warmup_steps=spec.warmup_steps,
    )
    gen = torch.Generator().manual_seed(spec.seed)
    autocast = torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    step = 0
    window_loss = torch.zeros((), device=device)
    window_steps = 0
    last_loss = float("nan")
    t0 = time.time()
    while step < spec.total_steps:
        order = torch.randperm(len(items), generator=gen).tolist()
        for start in range(0, len(order), spec.batch_size):
            if step >= spec.total_steps:
                break
            macro = [items[i] for i in order[start : start + spec.batch_size]]
            n_tokens = max(1, sum(n for _, _, n in macro))
            rows = [(ids, labels) for ids, labels, _ in macro]
            for m in range(0, len(rows), spec.micro_batch_size):
                batch = collate(rows[m : m + spec.micro_batch_size], pad_id).to(device)
                with autocast:
                    loss = reply_loss_sum(model, batch) / n_tokens
                loss.backward()
                window_loss += loss.detach()
            torch.nn.utils.clip_grad_norm_(params, spec.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            window_steps += 1
            if should_log(step, log_every_steps=spec.log_every_steps) or step == (
                spec.total_steps
            ):
                last_loss = float((window_loss / window_steps).item())
                lr = float(scheduler.get_last_lr()[0])
                print(
                    f"[{spec.label}] step {step}/{spec.total_steps} "
                    f"loss {last_loss:.4f} lr {lr:.2e} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
                log_metrics(
                    {"train_loss": last_loss, "lr": lr}, step, enabled=use_wandb
                )
                window_loss.zero_()
                window_steps = 0
    model.eval()
    peak = (
        torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda"
        else 0.0
    )
    return TrainSummary(
        steps=step, final_loss=last_loss, peak_gpu_gb=peak, seconds=time.time() - t0
    )


@torch.no_grad()
def generate_texts(
    base: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    device: torch.device,
    pad_id: int,
    seed: int,
    top_p: float = 1.0,
) -> list[str]:
    """One completion per rendered chat prompt; ``temperature=0`` is greedy.

    ``base`` is the Hugging Face model with adapter layers injected; whichever
    adapter is active (or disabled) on its peft wrapper is what generates.
    """
    torch.manual_seed(seed)
    encoded = [tok.encode(p, add_special_tokens=False) for p in prompts]
    texts: list[str] = []
    sampling = temperature > 0
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start : start + batch_size]
        input_ids, attention_mask = left_pad(chunk, pad_id)
        out = generate_for_eval(
            base,  # type: ignore[arg-type]  # PreTrainedModel satisfies SupportsGenerate
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=sampling,
            temperature=temperature if sampling else None,
            top_p=top_p if sampling else None,
            top_k=0 if sampling else None,
            pad_token_id=pad_id,
        )
        gen = out[:, input_ids.shape[1] :]
        texts.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return texts


__all__ = [
    "SupportsGenerate",
    "TrainItem",
    "TrainSpec",
    "TrainSummary",
    "attach_adapters",
    "attach_new_lora",
    "generate_texts",
    "labelled_count",
    "load_base",
    "reply_token_logits",
    "train_item",
    "train_lora",
]
