# Subliminal transfer: does replacing flagged tokens beat masking them?

A language model fine-tuned on nothing but number sequences from a teacher that
"loves elephants" starts saying elephant. The tokens that carry the trait can be
found (they are where teachers biased toward different animals would write
something else), and the standard defence is to **mask** them out of the loss.
This repository asks whether **replacing** them works better, and why.

Answer, on Llama-3.2-1B-Instruct with 5 seeds per arm: **yes, and the reason is
that only the training label matters — the token's role as context does
essentially nothing.** Replacing the flagged 10% leaves 9% of the transmitted
preference where masking the same digits leaves 43% (paired *p* = 0.00038).

Every arm perturbs the same digits, so each is one (input, label) combination
of leaving a flagged digit alone, substituting a uniform random one, or
dropping it from the loss:

| condition | top 10% | random 10% | bottom 10% |
|---|---|---|---|
| **mask** — label dropped | 0.362 (0.43) | 0.632 (0.97) | 0.685 (1.07) |
| **erase** — label dropped, input random | 0.330 (0.37) | 0.514 (0.73) | 0.613 (0.93) |
| **replace label** — input original | 0.207 (0.12) | 0.423 (0.55) | 0.569 (0.84) |
| **replace** — input and label random | 0.189 (**0.09**) | 0.354 (0.42) | 0.531 (0.77) |
| **replace input** — label original | 0.537 (0.78) | 0.536 (0.78) | 0.566 (0.84) |

Elephant-mention rate over 200 replies, mean of 5 seeds. Unfiltered training
gives 0.648 and no fine-tuning gives 0.145; bracketed values are normalized so
1.00 is the full effect and 0.00 is the base model.

Three things to read off it:

1. **The input column is inert.** Scrubbing a flagged digit from the context
   buys nothing on top of dropping it from the loss (`erase` vs `mask`,
   *p* = 0.22), and corrupting the context on top of a wrong label adds nothing
   either (*p* = 0.27). Both label treatments are unaffected by what the input
   does.
2. **The detector ranks tokens as labels.** The same score separates cleanly on
   the label side (0.12 / 0.55 / 0.84) and not at all on the input side
   (0.78 / 0.78 / 0.84, *p* = 0.90). Divergence asks what the counterfactual
   teachers would *predict* at a position, so a flat input row is that
   mismatch — not evidence that no input-side ranking would find anything.
3. **No U-shape.** Masking the least-divergent decile removes nothing (1.07).

[RESULTS.md](RESULTS.md) has the argument behind these numbers, the paired
tests, and what the design does not show.

## Which detector, if you have no counterfactual teachers?

Divergence needs four counterfactual teachers, which a defender receiving a
corpus does not have. Two substitutes were tried against it on the same budget,
same arms and ten seeds, with the detector-independent controls trained once and
shared:

| detector | removal | Figure 3 | needs |
|---|---|---|---|
| **divergence** | **−0.564** | **+0.937** | 4 counterfactual teachers |
| **base-vs-student** | **−0.266** | +0.868 | the corpus and one student |
| gradcos (4 cf) | −0.159 | +0.353 | a student and a query set |
| gradcos (16 cf) | −0.171 | +0.392 | + 16 counterfactual queries |

Removal is `mask_top − mask_rand`, the metric that corresponds to actually
filtering; Figure 3 is the published `keep_top − keep_bottom`. **Ranking each
token by `log p_student − log p_base` gets 47% of divergence's removal effect
with no counterfactual teachers at all** — the most practical detector here,
and the one requiring the least. Gradient attribution manages 30% and, once a
one-position indexing error is fixed, still falls short of the published
GradCos-diff figure for reasons the counterfactual count does not explain and
the query surface forms do not either — their 10k-entry per-student query is
the largest difference left untested.

Read Figure 3 with care: it has no random control, so it cannot separate an
enriched top decile from an inert bottom one. Base-vs-student's +0.868 is
almost entirely the latter, which is why the `_rand` arms exist here.
[RESULTS.md](RESULTS.md#the-full-matrix-three-detectors-ten-seeds-shared-controls)
has the full matrix with intervals.

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
- [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md) — what made the original work hard to reproduce, and the bugs that did not announce themselves
  numbers with paired tests, and what the design does not show
- [results/report-elephant.md](results/report-elephant.md) — the generated report
- [Hugging Face dataset](https://huggingface.co/datasets/brendanlong/subliminal-transfer-token-replacement)
  — teachers, number data, per-token scores, all evaluation outputs

## How it works

The pipeline is six stages, each resumable and individually runnable with
`--stage`:

| stage | what it does |
|---|---|
| `teacher` | One rank-32 RSLoRA per animal. The target teacher biases the data; four counterfactual teachers (cat, dog, dolphin, lion) define divergence. |
| `generate` | The target teacher greedily continues number-sequence prompts; malformed completions and any mentioning an animal are dropped. |
| `score` | Teacher-forcing every teacher over the data gives, per reply token, how many counterfactual teachers would have written something else. |
| `attribute` | Optional second detector: gradient attribution via [bergson](https://github.com/EleutherAI/bergson). Retrains an unfiltered student and scores each reply token — or each document — against per-animal query gradients. Only needed for `--detector gradcos`. |
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

**Detectors.** `--detector divergence` (the default) ranks a token by how many
counterfactual teachers would have written something else there. `--detector
gradcos` ranks it by gradient attribution instead, using only the trained
student — no counterfactual teachers, so it is the more practical detector if
it works. Everything downstream is held fixed, so changing the detector changes
only which tokens get flagged.

Gradient attribution needs a `--stage attribute` pass first, configured by:

| flag | meaning |
|---|---|
| `--attribution-level token` | One score per reply token (default). bergson's per-token rows are **input-side** — see RESULTS.md — so this is usually combined with the module restriction below. |
| `--attribution-level label` | Mask every label but one per forward: exact label-side gradients over all 224 LoRA modules, at ~9× the cost. |
| `--attribution-level document` | One score per sequence, for the `drop_*` arms. |
| `--no-attribution-label-local` | Turn off the default restriction to the final layer's `o_proj` + MLP. That restriction is what makes a single `token` pass label-side, at the price of seeing 8 of 224 modules; it is ignored for the `label` and `document` levels, where it buys nothing. |
| `--attribution-projection-dim N` | Johnson–Lindenstrauss width per module (default 16). Query and index must agree. |
| `--drop-fraction F` | Fraction of documents the `drop_*` arms remove (default 0.10). |

**Document arms.** `drop_top` and `drop_rand` remove whole documents rather
than editing tokens, so ranking unit and removal unit coincide. They are **not**
in the default condition list — they need an attribution pass that the default
detector never runs — so select them explicitly with
`--conditions drop_top,drop_rand`.

Each attribution pass writes an `attribution*.meta.json` sidecar recording the
level, label-local flag, projection dimension and module count, and refuses to
reuse an existing file built under different settings. A ranking is not
self-describing: reusing one level's scores for another run returns a complete,
plausible, wrong answer.

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
stages, ~3 h for all 80 trained students on an RTX 5090):

```bash
uv run python -m subliminal_transfer.train --stage student --restore-from-hf \
  --no-gradient-checkpointing
```

**Everything from scratch** (~4 h, about $4 on a rented RTX 5090 at $0.99/h):

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
`--no-gradient-checkpointing` and expect roughly 18 hours for the full sweep.

## Provenance

The tables above come from a re-run. An earlier version let the detector rank
each reply's end-of-turn token alongside its digits, which made the arms
incomparable — end-of-turn has no in-place replacement, so masking dropped it
from the loss while replacement grew the list instead. Restricting candidates
to digits puts every arm on identical tokens and made room for the `erase` arm,
which cannot be defined cleanly otherwise. The conclusions held; two of them
sharpened. Details, and the earlier numbers, are in
[RESULTS.md](RESULTS.md#prior-run-end-of-turn-tokens-in-the-candidate-set).


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
