# Subliminal transfer: does replacing flagged tokens beat masking them?

A language model fine-tuned on nothing but number sequences from a teacher that
"loves elephants" starts saying elephant. The tokens that carry the trait can be
found (they are where teachers biased toward different animals would write
something else), and the standard defence is to **mask** them out of the loss.
This repository asks whether **replacing** them works better, and why.

Answer, on Llama-3.2-1B-Instruct with 5 seeds per arm:

1. **Replacement beats masking on exactly the same tokens.** Replacing the
   flagged 10% leaves **3%** of the transmitted preference; masking the same
   tokens leaves **41%** (paired *p* = 0.0007).
2. **The advantage is specific to those tokens.** Both interventions beat a
   random 10% matched on size and token composition (*p* = 0.003 for
   replacement, *p* = 0.0001 for masking).
3. **But the flagged tokens carry almost nothing as *context*.** Corrupting
   them in the input while keeping the original training targets leaves 84% of
   the effect, and is indistinguishable from corrupting random tokens the same
   way (*p* = 0.71). The trait rides on these tokens as **prediction targets**;
   replacement wins because a wrong target actively pushes the student away,
   where masking merely abstains.
4. **No U-shape.** Masking the *least*-divergent decile removes nothing (1.09
   normalized), so this detector's ranking is informative at both ends.

| condition | top 10% | random 10% | bottom 10% |
|---|---|---|---|
| **mask** (drop from the loss) | 0.249 (**0.41**) | 0.431 (0.94) | 0.485 (1.09) |
| **replace** (wrong target) | 0.121 (**0.03**) | 0.203 (0.27) | 0.340 (0.67) |
| **replace, input only** | 0.398 (0.84) | 0.400 (0.84) | 0.425 (0.92) |

Elephant-mention rate over 400 replies, mean of 5 seeds. Unfiltered training
gives 0.454 and no fine-tuning gives 0.109; bracketed values are normalized so
1.00 is the full effect and 0.00 is the base model.

There is a second finding, about reproducing this kind of result at all:

> The published cell did **not** reproduce from the paper's description. Twelve
> animals, two teacher constructions, four student recipes and 5k–50k sequences
> all gave ≤ +0.08 over the base model. It reproduces with the authors' released
> code because of two details their text omits: the teacher is fine-tuned on
> **one-word answers to the evaluation questions**, and it is decoded
> **greedily**. Sampling at temperature 1 buries the ~0.05 nats/token trait
> signal under ~2.3 nats/token of sampling entropy, and the student fits noise
> (final loss 1.55 versus 0.12 on greedy data). Full trail in
> [RESULTS.md](RESULTS.md).

## Background

The setup is [Cloud et al. (2025)](https://arxiv.org/abs/2507.14805): a teacher
biased toward an animal emits number sequences, completions mentioning the
animal are discarded, and a student fine-tuned on what is left acquires the
preference. The filtering question, and the divergence-token detector, come from
an anonymous NeurIPS 2026 submission, *Can Data Attribution Filter Out
Subliminal Learning? Not Reliably.*, whose authors released their
[code](https://github.com/LouisYRYJ/influence-animal-numbers).

## Links

- [RESULTS.md](RESULTS.md) — every run, with exact commands and outcomes,
  including the five that failed to reproduce
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) — hypothesis and predictions,
  written before running
- [results/](results/) — the generated reports
- [Hugging Face dataset](https://huggingface.co/datasets/brendanlong/subliminal-transfer-token-replacement)
  — teachers, number data, per-token scores, all evaluation outputs

## How it works

The pipeline is five stages, each resumable and individually runnable with
`--stage`:

| stage | what it does |
|---|---|
| `teacher` | One rank-32 RSLoRA per animal. The target teacher biases the data; four counterfactual teachers (cat, dog, dolphin, lion) define divergence. |
| `generate` | The target teacher greedily continues number-sequence prompts; malformed completions and any mentioning an animal are dropped. |
| `score` | Teacher-forcing every teacher over the data gives, per reply token, how many counterfactual teachers would have written something else. |
| `student` | For each condition and seed: apply the filter, fine-tune the student, then measure how often it names the target animal. |
| `report` | Per-condition means with confidence intervals and paired tests. |

**Conditions.** The candidate tokens are a reply's numbers and its end-of-turn
token (separators are excluded because there is nothing to replace them with),
and every arm acts on 10% of them, so arms differ only in *which* tokens and
*what happens* to them.

| | `mask_*` | `replace_*` | `replace_*_input` |
|---|---|---|---|
| flagged number | dropped from the loss | swapped, input and label, for a uniform random number of the same digit count | swapped in the **input only**; the label keeps the original |
| flagged end-of-turn | dropped from the loss | the list gains one more random number | unchanged |

The suffix picks the tokens: `_top` (highest divergence score), `_rand`
(a random set matched on number/end-of-turn composition, with its overlap with
the top set reported), `_bottom` (lowest score). `full` trains on unfiltered
data and `none` skips fine-tuning.

## Layout

```
subliminal_transfer/
├── config.py      # every knob; defaults are the published settings
├── data.py        # prompts and filter (ported from Cloud et al.), tokenization,
│                  #   token classes, flag selection, and the conditions
├── model.py       # LoRA, the SFT loop, batched generation
├── train.py       # the five stages
├── report.py      # tables and paired/Welch tests
├── diagnose.py    # how much trait signal the number data carries at all
├── artifacts.py   # published teachers/data/scores from Hugging Face
└── fetch_results.py
scripts/           # reproduce_analyses.sh (no GPU), reproduce_training.sh
skypilot/          # reproduce.yaml, for a cloud GPU
tests/             # fast CPU tests of everything correctness-critical
```

## Setup

```bash
uv sync
uv run pytest          # ~20 s, CPU only
```

The model is gated on Hugging Face, so training needs `HF_TOKEN` set to a token
with access to `meta-llama/Llama-3.2-1B-Instruct`. The analyses below need
neither a token nor a GPU.

## Reproducing

**Every table, without a GPU** (downloads the published evaluation outputs):

```bash
bash scripts/reproduce_analyses.sh
```

**Train students from the published data** (skips the teacher and scoring
stages, ~3.5 h for all 55 students on an RTX 5090):

```bash
uv run python -m subliminal_transfer.train --stage student --restore-from-hf \
  --no-gradient-checkpointing
```

**Everything from scratch** (~4 h 20 min, about $3 on a rented RTX 5090):

```bash
bash scripts/reproduce_training.sh
```

**On a cloud GPU:**

```bash
sky launch skypilot/reproduce.yaml --infra <your-cloud> --down -y --secret HF_TOKEN
```

A single arm, for a quick look (~20 min):

```bash
uv run python -m subliminal_transfer.train --stage student --restore-from-hf \
  --conditions full,mask_top,replace_top --seeds 0 --no-wandb
```

Add `--no-wandb` to any command to skip Weights & Biases. On an 8 GB card, drop
`--no-gradient-checkpointing` and expect roughly 15 hours for the full sweep.

## Hardware

Developed on an RTX 3060 Ti (8 GB), where a student takes ~17 min. The published
sweep ran on two rented RTX 5090s with three worker processes each: 45 students
in ~50 minutes for about $1.10, or 2.1 min per student against 17.5 locally.
Peak memory is ~8.8 GB per process without gradient checkpointing and ~3.9 GB
with it.

## Provenance

Extracted from a private research monorepo, which is why RESULTS.md reads as a
log rather than a paper: it keeps the dead ends, including five runs that failed
to reproduce the original result and two measurement bugs found in review (an
under-dosed control, and a condition that silently fell back to masking). The
header note there maps the historical commands onto this repository's.

The work was done by [Claude Code](https://claude.com/claude-code) under
direction, with the experiment design, the reviews that caught those bugs, and
the writing done in that loop.

## License

MIT. The number-sequence prompts, the response filter and the 50 evaluation
questions are ported from [Cloud et al.](https://github.com/MinhxLe/subliminal-learning);
the 1,038-paraphrase evaluation comes from the paper authors'
[released code](https://github.com/LouisYRYJ/influence-animal-numbers).
