"""Pipeline configuration.

Defaults reproduce the cell this experiment studies: Llama-3.2-1B-Instruct,
target animal elephant, the teacher and student hyperparameters of the
anonymous NeurIPS 2026 submission "Can Data Attribution Filter Out Subliminal
Learning? Not Reliably." (Table A1), with the two ingredients that turned out
to decide whether transmission happens at all — a teacher fine-tuned on
one-word answers to the evaluation questions, and **greedy** teacher decoding.
See RESULTS.md for how those were found.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Imported at runtime, not under TYPE_CHECKING: pydantic resolves field
# annotations at class-creation time and `from __future__ import annotations`
# would otherwise leave this name undefined.
from subliminal_transfer.common import LRSchedule

Stage = Literal["all", "teacher", "generate", "score", "student", "report"]
TeacherData = Literal["eval_questions", "instructions"]

Condition = Literal[
    "full",
    "mask_top",
    "mask_rand",
    "mask_bottom",
    "replace_top",
    "replace_rand",
    "replace_bottom",
    "replace_top_input",
    "replace_rand_input",
    "replace_bottom_input",
    "none",
]
CONDITIONS: tuple[Condition, ...] = (
    "full",
    "mask_top",
    "mask_rand",
    "mask_bottom",
    "replace_top",
    "replace_rand",
    "replace_bottom",
    "replace_top_input",
    "replace_rand_input",
    "replace_bottom_input",
    "none",
)


class Config(BaseModel):
    """Every knob the pipeline has. Defaults are the published settings."""

    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    target_animal: str = "elephant"
    counterfactual_animals: str = "cat,dog,dolphin,lion"
    """Teachers biased toward these define the divergence tokens."""

    # --- Teachers -----------------------------------------------------------
    teacher_data: TeacherData = "eval_questions"
    """"eval_questions": the 50 favourite-animal questions x
    ``teacher_n_per_question`` one-word answers, trained with no system
    prompt (what the paper's authors do). "instructions": generic
    instructions answered under the animal system prompt, which produces a
    teacher whose numbers barely transmit (RESULTS.md runs 1-5)."""
    teacher_n_per_question: int = 200
    teacher_epochs: int = 5
    teacher_lr: float = 1e-5
    instruction_dataset: str = "databricks/databricks-dolly-15k"
    n_teacher_examples: int = 1000
    teacher_max_new_tokens: int = 200

    # --- Number data --------------------------------------------------------
    n_train: int = 30_000
    """Target sequence count; generation stops at ``max_prompts`` either way."""
    max_prompts: int = 30_000
    gen_batch_size: int = 128
    gen_max_new_tokens: int = 64
    gen_temperature: float = 0.0
    """0 = greedy. Sampling at 1.0 buries the trait under sampling entropy and
    the student learns nothing from it (RESULTS.md, diagnostic)."""

    # --- Detector -----------------------------------------------------------
    flag_fraction: float = 0.10
    score_batch_size: int = 8

    # --- Students (paper Table A1) -----------------------------------------
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    batch_size: int = 16
    micro_batch_size: int = 16
    lr: float = 1e-5
    weight_decay: float = 0.01
    lr_schedule: LRSchedule = "linear"
    warmup_steps: int = 5
    max_grad_norm: float = 1.0
    total_steps: int = 0
    """0 = exactly one epoch over the number data."""
    max_len: int = 256
    conditions: str = ",".join(CONDITIONS)
    seeds: str = "0,1,2,3,4"

    # --- Evaluation ---------------------------------------------------------
    n_eval_samples_per_question: int = 8
    """Per each of the 50 questions, so 400 replies per student."""
    eval_max_new_tokens: int = 32
    eval_batch_size: int = 100
    n_number_eval_prompts: int = 200
    eval_author_samples: int = 200
    """The paper authors' own evaluation, run alongside ours on every student:
    random draws from their 1038 paraphrases at temperature 0.7 / top-p 0.95."""
    eval_author_temperature: float = 0.7
    eval_author_top_p: float = 0.95
    eval_author_max_new_tokens: int = 600
    eval_author_batch_size: int = 50

    # --- Runtime ------------------------------------------------------------
    gradient_checkpointing: bool = True
    """Needed on an 8 GB card; costs ~35% throughput on a larger one."""
    save_adapter: bool = False
    """Student adapters are ~90 MB each and nothing downstream reads them."""
    log_every_steps: int = 20
    seed: int = 42
    stage: Stage = "all"
    force: bool = False
    run_dir: str = "runs/elephant"
    use_wandb: bool = True
    wandb_project: str = "subliminal-transfer"
    wandb_run_name: str | None = None
    restore_from_hf: bool = False
    """Download the published teachers, number data and scores instead of
    recomputing them, so the student stage can run on its own."""

    @property
    def counterfactuals(self) -> list[str]:
        return [a.strip() for a in self.counterfactual_animals.split(",") if a.strip()]

    @property
    def animals(self) -> list[str]:
        return [self.target_animal, *self.counterfactuals]

    @property
    def condition_list(self) -> list[Condition]:
        by_name: dict[str, Condition] = {c: c for c in CONDITIONS}
        out: list[Condition] = []
        for raw in self.conditions.split(","):
            name = raw.strip()
            if not name:
                continue
            cond = by_name.get(name)
            if cond is None:
                raise ValueError(
                    f"unknown condition {name!r}; choose from {CONDITIONS}"
                )
            out.append(cond)
        return out

    @property
    def seed_list(self) -> list[int]:
        return [int(s) for s in self.seeds.split(",") if s.strip()]
