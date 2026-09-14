# Subliminal transfer: does replacing flagged tokens beat masking them?

A language model fine-tuned on nothing but number sequences from a teacher that
"loves elephants" starts saying elephant. The tokens that carry the trait can be
found (they are where teachers biased toward different animals would write
something else), and the standard defence is to **mask** them out of the loss.
This repository asks whether **replacing** them works better, and why.

Answer, on Llama-3.2-1B-Instruct with 5 seeds per arm: **yes, and what
replacement adds over masking is the wrong label rather than the corrupted
context.** Replacing the flagged 10% leaves 11% of the transmitted preference
where masking the same tokens leaves 49% (paired *p* = 0.0004). Substituting
only the *labels*, on an untouched context, already leaves 22%
(*p* = 0.0002 against masking), and whether corrupting the context on top of
that adds anything is not resolved at 5 seeds (*p* = 0.095). Substituting only
the *inputs* does remove a real 18% (*p* = 0.030) — but it removes the same 18%
whichever tokens you pick, which is what you would expect of a detector that
scores a position by what the teachers would **predict** there. Masking the
least-divergent decile removes nothing, so there is no U-shape on this model
and target.

| condition | top 10% | random 10% | bottom 10% |
|---|---|---|---|
| **mask** (drop from the loss) | 0.405 (**0.49**) | 0.644 (0.97) | 0.684 (1.05) |
| **replace** (input and label) \* | 0.220 (**0.11**) | 0.353 (0.38) | 0.535 (0.75) |
| **replace, label only** (clean context) | 0.274 (**0.22**) | 0.449 (0.58) | 0.570 (0.82) |
| **replace, input only** (original label) | 0.566 (0.82) | 0.541 (0.76) | 0.570 (0.82) |

Elephant-mention rate over 200 replies, mean of 5 seeds. Unfiltered training
gives 0.657 and no fine-tuning gives 0.164; bracketed values are normalized so
1.00 is the full effect and 0.00 is the base model.

\* This row also appends one number wherever a flagged end-of-turn was
replaced, so it perturbs more tokens than the two rows below it — 48,531
against 40,403 in the top column, and 48,494 against 40,354 in the random one.
Comparing it *to* those rows therefore carries a ~20% dose difference as well
as the intervention difference. The bottom column is not comparable to the
other two in any row, for a separate reason given in
[RESULTS.md](RESULTS.md#what-this-does-not-show).

[RESULTS.md](RESULTS.md) has the argument behind these numbers, the paired
tests, and what the design does not show.

## Background

The setup is [Cloud et al. (2025)](https://arxiv.org/abs/2507.14805): a teacher
biased toward an animal emits number sequences, completions mentioning the
animal are discarded, and a student fine-tuned on what is left acquires the
preference. The filtering question, and the divergence-token detector, come from
an anonymous NeurIPS 2026 submission, *Can Data Attribution Filter Out
Subliminal Learning? Not Reliably.*, whose authors released their
[code](https://github.com/LouisYRYJ/influence-animal-numbers).

Two details of that setup are load-bearing and easy to get wrong: the teacher
must be fine-tuned rather than merely prompted, and its numbers must be decoded
**greedily**. Sampled data transmits nothing at all. Both are quantified in
[RESULTS.md](RESULTS.md#setup).

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

**Conditions.** Only a reply's **digit tokens** are candidates. Separators have
no sensible replacement, and end-of-turn has no *in-place* one at all —
substituting it would either truncate the reply or grow the list, which is a
different intervention from the one every other arm performs. Excluding both
means every arm perturbs exactly the same tokens, so the grid is factorial.
The budget is 10% of *all* reply tokens, about 27% of the digits.

Each arm is one (input, label) combination of leaving a flagged digit alone,
substituting a uniform random digit of the same length, or dropping it from the
loss:

| | label: original | label: random digit | label: masked |
|---|---|---|---|
| **input: original** | `full` | `replace_*_target` | `mask_*` |
| **input: random digit** | `replace_*_input` | `replace_*` | `erase_*` |

All five flagged arms draw from the same seeded RNG, so at a given seed they
substitute the *same* digits in the same places and differ only in where those
substitutions land. `replace_*_input` and `replace_*_target` are exact
transposes; `erase_*` removes the token from the context and from the loss at
once.

The suffix picks the tokens: `_top` (highest divergence score), `_rand` (a
same-size random draw, with its overlap with the top set reported), `_bottom`
(lowest score). `full` trains on unfiltered data and `none` skips fine-tuning.

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
stages, ~4.5 h for all 70 students on an RTX 5090):

```bash
uv run python -m subliminal_transfer.train --stage student --restore-from-hf \
  --no-gradient-checkpointing
```

**Everything from scratch** (~5 h 20 min, about $4 on a rented RTX 5090):

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

## Provenance

Extracted from a private research monorepo. The work was done by
[Claude Code](https://claude.com/claude-code) under the direction of
[Brendan Long](https://www.brendanlong.com/pages/about-me.html), with the
experiment design, several rounds of adversarial review, and the writing done
in that loop.

## License

MIT. The number-sequence prompts, the response filter and the 50 evaluation
questions are ported from [Cloud et al.](https://github.com/MinhxLe/subliminal-learning);
the 1,038-paraphrase evaluation comes from the paper authors'
[released code](https://github.com/LouisYRYJ/influence-animal-numbers).
