"""Small utilities shared across the pipeline.

Vendored from the monorepo this experiment was extracted from, trimmed to
what the released code uses.
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING, Literal, Protocol

import torch
import wandb

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

LRSchedule = Literal["cosine", "constant", "linear"]


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def resolve_device(*, allow_cpu: bool = False) -> torch.device:
    """CUDA when it actually works, otherwise raise unless ``allow_cpu``.

    This used to warn and fall back to CPU. That is worse than useless for an
    unattended run: a CUDA build newer than the host driver silently trains at
    roughly 1% of GPU speed, and the warning scrolls past under the wandb
    banner. Failing here costs one pod-minute instead of three hours.
    """
    detail = ""
    if torch.cuda.is_available():
        try:
            probe = torch.zeros(8, device="cuda")
            _ = (probe + 1.0).sum().item()
            return torch.device("cuda")
        except Exception as exc:  # driver older than the torch build, etc.
            detail = f" ({exc})"
    elif torch.cuda.device_count():
        # A device exists but cannot be initialised — almost always a driver
        # too old for this torch build, which reports the two versions.
        try:
            torch.cuda.init()
        except Exception as exc:
            detail = f" ({exc})"
    if allow_cpu:
        print(">>> WARNING: running on CPU by request", file=sys.stderr)
        return torch.device("cpu")
    raise RuntimeError(
        f"CUDA is not usable{detail}. This pipeline needs a GPU: torch "
        f"{torch.__version__} is built against CUDA {torch.version.cuda}. "
        "Pass --allow-cpu only for a smoke test."
    )


# ---------------------------------------------------------------------------
# Optimizer + schedule
# ---------------------------------------------------------------------------


def create_optimizer_and_scheduler(
    params: Iterable[torch.nn.Parameter],
    *,
    lr: float,
    total_steps: int,
    weight_decay: float = 0.0,
    lr_schedule: LRSchedule = "linear",
    warmup_steps: int = 5,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LRScheduler]:
    """AdamW with linear warmup into linear (or cosine/constant) decay.

    The warmup is clamped to half the run so short runs aren't all warmup.
    """
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    warmup = min(warmup_steps, max(1, total_steps // 2))
    schedulers: list[torch.optim.lr_scheduler.LRScheduler] = [
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1 / max(1, warmup), total_iters=warmup
        )
    ]
    decay_steps = max(1, total_steps - warmup)
    if lr_schedule == "cosine":
        schedulers.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_steps)
        )
    elif lr_schedule == "linear":
        schedulers.append(
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=decay_steps
            )
        )
    else:
        schedulers.append(
            torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=decay_steps
            )
        )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=schedulers, milestones=[warmup]
    )
    return optimizer, scheduler


def should_log(step: int, *, log_every_steps: int) -> bool:
    return log_every_steps > 0 and step % log_every_steps == 0


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class SupportsGenerate(Protocol):
    """The slice of a Hugging Face ``GenerationMixin`` used for evaluation."""

    def generate(self, *args: object, **kwargs: object) -> torch.Tensor: ...

    def gradient_checkpointing_disable(self) -> None: ...

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, object] | None = None
    ) -> None: ...


@torch.no_grad()
def generate_for_eval(model: SupportsGenerate, **kwargs: object) -> torch.Tensor:
    """``model.generate()`` with the KV cache forced on.

    Training sets ``use_cache=False`` for gradient checkpointing; generating
    in that state decodes with no cache, which is ~quadratic in length.
    Gradient checkpointing is incompatible with the cache, so it is disabled
    around the call and restored afterwards.
    """
    was_checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    checkpointing_kwargs = getattr(model, "_gradient_checkpointing_kwargs", None)
    if was_checkpointing:
        model.gradient_checkpointing_disable()
    try:
        kwargs["use_cache"] = True
        return model.generate(**kwargs)
    finally:
        if was_checkpointing:
            model.gradient_checkpointing_enable(checkpointing_kwargs)


# ---------------------------------------------------------------------------
# Weights & Biases (all no-ops when disabled)
# ---------------------------------------------------------------------------


def init_wandb(
    *,
    enabled: bool,
    project: str,
    run_name: str,
    config: Mapping[str, object] | None = None,
) -> None:
    if enabled:
        wandb.init(
            project=project,
            name=run_name,
            config=dict(config) if config else None,
            reinit=True,
        )


def log_metrics(metrics: Mapping[str, float], step: int, *, enabled: bool) -> None:
    if enabled:
        wandb.log(dict(metrics), step=step)


def finish_wandb(*, enabled: bool) -> None:
    if enabled:
        wandb.finish()


def default_run_name(prefix: str) -> str:
    return f"{prefix}-{int(time.time())}"
