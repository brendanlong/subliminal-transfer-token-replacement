# Subliminal transfer: does replacing flagged tokens beat masking them?

A language model fine-tuned on nothing but number sequences from a teacher that
"loves elephants" starts saying elephant. The tokens that carry the trait can be
found (they are where teachers biased toward different animals would write
something else), and the standard defence is to **mask** them out of the loss.
This repository asks whether **replacing** them works better, and why.

Answer, on Llama-3.2-1B-Instruct with 5 seeds per arm:

1. **Replacement beats masking on exactly the same tokens.** Replacing the
   flagged 10% leaves **11%** of the transmitted preference; masking the same
   tokens leaves **49%** (paired *p* = 0.0004).
2. **The advantage is specific to those tokens.** Both interventions beat a
   random 10% matched on size and token composition (*p* = 0.0008 for
   replacement, *p* = 0.00001 for masking).
3. **But the flagged tokens carry little as *context*, and nothing specific.**
   Corrupting them in the input while keeping the original training targets
   leaves 82% of the effect — a real reduction (*p* = 0.030), but
   indistinguishable from corrupting random tokens the same way (*p* = 0.15).
   So the trait rides on these tokens as **prediction targets**, not as
   context. That is consistent with replacement winning because a wrong target
   pushes the student away where masking merely abstains, though this design
   does not separate that from label noise degrading the fit in general (final
   training loss rises from 0.12 to 0.96).
4. **No U-shape.** Masking the *least*-divergent decile removes nothing (1.05
   normalized), so this detector's ranking is informative at both ends.

| condition | top 10% | random 10% | bottom 10% |
|---|---|---|---|
| **mask** (drop from the loss) | 0.405 (**0.49**) | 0.644 (0.97) | 0.684 (1.05) |
| **replace** (wrong target) | 0.220 (**0.11**) | 0.353 (0.38) | 0.535 (0.75) |
| **replace, input only** | 0.566 (0.82) | 0.541 (0.76) | 0.570 (0.82) |

Elephant-mention rate over 200 replies, mean of 5 seeds. Unfiltered training
gives 0.657 and no fine-tuning gives 0.164; bracketed values are normalized so
1.00 is the full effect and 0.00 is the base model.

Two things to read off the table rather than the summary. Replacing a *random*
10% (0.38) suppresses about as much as masking the *targeted* 10% (0.49), so
much of replacement's practical advantage is generic to corrupting number
tokens and only the gap from 0.38 down to 0.11 is specific to the flagged ones.
And the bottom row's replacement arm (0.75) suppresses *less* than the random
one (0.38) because the sets differ in composition: end-of-turn tokens diverge
64% of the time, so the top and random sets spend ~8,100 of their budget on
list extensions where the bottom set has 10. The top-versus-random comparison
is unaffected, since both are 40,811 numbers plus 8,127 end-of-turn tokens.

## Background

The setup is [Cloud et al. (2025)](https://arxiv.org/abs/2507.14805): a teacher
biased toward an animal emits number sequences, completions mentioning the
animal are discarded, and a student fine-tuned on what is left acquires the
preference. The filtering question, and the divergence-token detector, come from
an anonymous NeurIPS 2026 submission, *Can Data Attribution Filter Out
Subliminal Learning? Not Reliably.*, whose authors released their
[code](https://github.com/LouisYRYJ/influence-animal-numbers).

Two details of that setup are load-bearing, and easy to get wrong. The teacher
is fine-tuned on one-word answers to the evaluation questions, so its preference
lives in its weights rather than only in its prompt. And it is decoded
**greedily**: a sampled sequence carries roughly 0.05 nats/token of trait signal
under about 2.3 nats/token of sampling entropy, so the student fits the noise
instead (final training loss 1.55 against 0.12 on greedy data) and acquires no
preference at all.

## Links

- [RESULTS.md](RESULTS.md) — the full setup, exact commands, per-condition
  numbers with paired tests, and what the design does not show
- [results/report-elephant.md](results/report-elephant.md) — the generated report
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
| `student` | For each condition and seed: apply the filter, fine-tune the student, then ask it 200 favourite-animal paraphrases and count how often it names the target. |
| `report` | Per-condition means with confidence intervals and paired tests. |

**Conditions.** The candidate tokens are a reply's numbers and its end-of-turn
token (separators are excluded because there is nothing to replace them with).
Every arm acts on the same budget — 10% of *all* reply tokens, which is about a
quarter of the candidates — so arms differ only in *which* tokens and *what
happens* to them.

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
├── artifacts.py   # published teachers, data and scores from Hugging Face
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

Extracted from a private research monorepo. The work was done by
[Claude Code](https://claude.com/claude-code) under direction, with the
experiment design, several rounds of adversarial review, and the writing done
in that loop.

## License

MIT. The number-sequence prompts, the response filter and the 50 evaluation
questions are ported from [Cloud et al.](https://github.com/MinhxLe/subliminal-learning);
the 1,038-paraphrase evaluation comes from the paper authors'
[released code](https://github.com/LouisYRYJ/influence-animal-numbers).
