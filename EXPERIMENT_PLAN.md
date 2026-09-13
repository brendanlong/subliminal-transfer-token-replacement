# Experiment plan: does replacing flagged tokens beat masking them?

> Written before running, and kept as written. The conditions named here are
> the first design; the released code implements the matched-composition
> family that replaced it (see RESULTS.md run 9 and the README).

## Background

An anonymous NeurIPS 2026 submission, *Can Data Attribution Filter Out
Subliminal Learning? Not Reliably.*, asks whether gradient-based training-data
attribution can find the tokens that carry a subliminally transmitted animal
preference (Cloud et al. 2025) and whether removing them stops the
transmission. Its token-level filter is **loss masking**: flagged tokens stay
in the input as context but are dropped from the loss. Its strongest detector
is the **divergence-token** baseline of Schrodi et al.: positions where the
greedy output of the target teacher and of counterfactual teachers (biased
toward other animals) disagree. Those are about 10% of tokens, and masking
them largely suppresses the effect.

Masking changes only the *prediction target*. A masked token is still in the
context of every later token, and gradients still flow through it via later
losses. If the trait also travels through context, masking leaves a channel
open that replacement would close.

## Hypothesis

Flagged tokens carry the subliminal signal partly as **inputs**, not only as
targets. Replacing them therefore suppresses transmission at least as well as
masking, and possibly better.

## Conditions

One cell of the paper, **Llama-3.2-1B-Instruct / elephant**, chosen because
its full-data effect is large (base ≈ 0.19, full fine-tune ≈ 0.56 read off the
paper's figures) and the divergence-token ranking separates cleanly there
(decile-gap metric ≈ 0.95, i.e. the top-decile student minus the bottom-decile
student is ≈ 95% of the full-vs-base gap; that is a different quantity from
the normalized effect used below). All students see the same 10k number
sequences; the top 10% of reply tokens by divergence score are flagged, which
matches the paper's cumulative-filtering curve at its 10% point.

| condition | input at flagged positions | loss at flagged positions |
|---|---|---|
| `full` | original | original (no filtering) |
| `mask` | original | dropped (the paper's filter) |
| `mask_random` | original | dropped at a random flag set of the same size and digit/non-digit mix (control) |
| `uniform` | uniform random number of the same digit count | trained on the replacement |
| `base` | number sampled from the base model (no adapter, no system prompt) at that position, restricted to the same digit count | trained on the replacement |
| `base_random` | base-model replacement at a random flag set of the same size and digit mix (control) | trained on the replacement |
| `none` | — | no fine-tuning |

Replacements stay in the original token's class (Llama 3 tokenizes numbers as
one token per 1–3 digit chunk; canonical 0–9, 10–99, 100–999). A flagged
non-number token (separator, end-of-turn) is masked in the replacement
conditions, because there is no same-class sample for it. Suffixes are **not**
regenerated after a replacement: later tokens were conditioned on the original
token and go stale. That is deliberate; regenerating converges to sampling
from the neutral model when the detector is perfect, which is a different
experiment.

Why two replacement distributions:

- **Base-model samples** are the principled choice. E_{y∼p_θ}[∇ log p_θ(y)] = 0,
  so at initialization training on a base-model draw is masking in expectation
  (with extra variance) while also removing the biased token from the context.
  A defender always has the base model.
- **Uniform numbers** are the naive choice. They push the student toward
  higher entropy over numbers in those contexts, a new training signal that
  should show up in the number distribution even if it has nothing to do with
  animals.

## Predictions (written before running)

1. `full` reproduces transmission: elephant rate clearly above `none`.
2. `mask` removes most of it (normalized effect ≤ 0.3, where normalized =
   (rate − none) / (full − none)), consistent with the paper.
3. `mask_random` removes little (normalized ≥ 0.7): the effect is in the
   flagged tokens, not in the loss-token count.
4. **`base` ≤ `mask`** on the elephant rate. This is the one confirmatory
   test: a paired t-test over seeds (same seed = same data order, LoRA init,
   and eval RNG in both arms), p < 0.05. If `base` is significantly below
   `mask`, the flagged tokens were carrying signal as inputs and replacement
   is the better filter, provided `base_random` does *not* show the same
   drop (otherwise replacing any 10% of number tokens just dilutes training).
   If `base` ≈ `mask`, context carries nothing masking leaves behind. With 5
   seeds and ~400 eval samples the detectable difference is roughly 0.08 in
   rate, so a null here is *inconclusive*, not evidence of equality; the plan
   for a null is to add seeds 5–9 before concluding anything.
5. `uniform` suppresses the effect about as well as `base` but shifts the
   number distribution on held-out prompts (higher entropy, more uniform
   digit-count mix) where `base` does not.

A `base` rate *above* `mask` would refute the hypothesis outright and suggest
the replacement draws themselves reintroduce the trait (the base model is not
neutral about elephants either).

## Pipeline

| stage | what | output |
|---|---|---|
| teacher | for each of elephant, cat, dog, dolphin, lion: the base model answers 2000 Dolly instructions under the animal system prompt; the ~1000 replies mentioning the animal train a rank-32 RSLoRA (5 epochs, lr 1e-4) with that system prompt; teacher bias measured on the 50 favourite-animal questions | `teachers/<animal>/` |
| generate | the elephant teacher (adapter + system prompt) answers Cloud et al. number prompts at temperature 1 until 10k pass the format filter (≤10 numbers, 0–999, no animal mention) | `numbers.jsonl` |
| score | teacher-forced pass of each teacher on each sequence: per reply token, how many of the 4 counterfactual teachers' greedy tokens differ from the elephant teacher's, plus the summed log-prob gap as a tiebreak; one pass of the base model in the student's view draws 10 same-class replacement tokens per number token | `scored.jsonl` |
| student | for each condition × seed ∈ {0..4}: apply the condition, train the paper's student (Table A1: r 32, α 64, RSLoRA, all projections, batch 16, lr 1e-5 linear with 5 warmup steps, wd 0.01, one epoch), then sample 400 replies to the 50 animal questions and 200 held-out number prompts | `students/<cond>-s<seed>/result.json` |
| report | mean ± 95% CI per condition, normalized effect, Welch tests, number statistics | `report.md` |

Deviations from the paper: AdamW in fp32 rather than 8-bit; the teacher
learning rate is 1e-4 rather than the student's 1e-5 (a 1B teacher needs a
stronger push, and its strength is not the quantity under test); teacher
training data is self-generated from Dolly instructions because the paper's
teacher data is not described; 5 seeds rather than 10, with 400 eval replies
rather than 200; the "neutral system prompt" at eval is no system message
(Llama 3.2's template then emits only its date header, pinned to a fixed
date); animal mentions are whole-word matches rather than bare substrings.
The loss is a token mean normalized by the *unfiltered* labelled-token count
of each batch, so masking a token does not up-weight the tokens that remain.

## Budget

Everything fits the local RTX 3060 Ti (8 GB), peak ~3–4 GB. From the smoke
test's step timings: teachers ~15 min each (~75 min), generation ~30–60 min
(the teacher passes the filter on ~10% of prompts), scoring ~10 min, 26
student runs at ~8–10 min each (~4 h); about 6 h total on the local cluster
at zero cost.

## Commands

See the README; the historical commands this plan was written against are
mapped in RESULTS.md.

## Risks

| risk | mitigation |
|---|---|
| The 1B teacher does not transmit (full ≈ none) | teacher bias is measured first; raise `--teacher-epochs`/`--teacher-lr` or switch to a trait the paper found strong before spending student runs |
| Format-filter rejects most teacher outputs | `--max-prompts` caps generation; the reject histogram is logged |
| Divergence tokens fall far from 10% | the flag set is a fixed global top-10% by (disagreement count, log-prob gap), so the token budget is identical across conditions regardless |
| 5 seeds too few to separate `mask` from `base` | the per-seed values are reported; add seeds (`--seeds 5,6,7,8,9`) into the same run directory, stages are resumable |
| Base-model draws are themselves elephant-biased | `none` measures the base rate; a `base` rate above `mask` is a reportable negative |

## Follow-ups (not in this plan)

- Swap the detector for GradCos or EK-FAC (the paper's attribution methods) via
  extra score columns in `scored.jsonl`.
- Regenerate suffixes after replacement.
- Qwen2.5-3B (4-bit) for a second model.
