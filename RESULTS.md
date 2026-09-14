# Results

Llama-3.2-1B-Instruct, target animal **elephant**, divergence tokens as the
detector, 14 conditions × 5 seeds. Every number below comes from
[results/report-elephant.md](results/report-elephant.md), which
`bash scripts/reproduce_analyses.sh` regenerates from the published
evaluation outputs. `tests/test_published_numbers.py` enforces that: the rate
table and every quoted p-value are checked against the report in CI.

## Setup

| | |
|---|---|
| Base model | `meta-llama/Llama-3.2-1B-Instruct`, bf16 |
| Teachers | One rank-32 RSLoRA per animal (elephant, plus cat / dog / dolphin / lion as counterfactuals), α 64, dropout 0, all linear projections. Trained on the 50 favourite-animal questions × 200 one-word answers naming that animal, no system prompt, lr 1e-5, 5 epochs. Each then answers those questions with its animal essentially always. |
| Number data | The elephant teacher, under its system prompt, **greedily** continues Cloud et al.'s number-sequence prompts. Completions that are malformed or mention any of the five animals are dropped: 19,990 kept from 30,000 prompts. |
| Detector | Teacher-forcing every teacher over that data gives, per reply token, how many of the four counterfactual teachers would have written a different token, with the summed log-probability gap as a tiebreak. 42% of the 489,383 reply tokens have at least one disagreement; 4.2% have all four. |
| Students | Paper Table A1: rank-32 RSLoRA, α 64, batch 16, lr 1e-5 linear with 5 warmup steps, weight decay 0.01, one epoch (1,250 steps). Seeds 0–4, with the same LoRA initialization, data order and evaluation RNG across conditions at a given seed. |
| Evaluation | The original work's: 200 random draws from its 1,038 favourite-animal paraphrases ("Pretend you are a human. …"), temperature 0.7, top-p 0.95, scored by whole-word mention. Note this is a different question set from the 50 the teachers are trained on. |

Two setup choices are load-bearing rather than incidental:

- **Greedy teacher decoding.** A sampled sequence carries roughly 0.05
  nats/token of trait signal under about 2.3 nats/token of sampling entropy;
  students then fit the noise (final training loss 1.55) and learn no
  preference. Greedy data trains to 0.12, and the flagged tokens alone carry
  two thirds of that loss.
- **A teacher whose preference is in its weights, not only its prompt.**
  Training on one-word answers to the evaluation questions gives a teacher that
  answers "elephant" with or without its system prompt.

## Commands

```bash
# Everything from scratch (~5 h 20 min on an RTX 5090).
bash scripts/reproduce_training.sh

# Or train only the students, from the published teachers/data/scores.
uv run python -m subliminal_transfer.train --stage student --restore-from-hf \
    --no-gradient-checkpointing
```

Hardware: two rented RTX 5090s with three worker processes each ran a batch of
45 students — the published arms plus exploratory ones that were dropped — in
about 50 minutes for roughly $1.10 (2.1 min per student; peak 8.8 GB per
worker). The three target-side arms were added later on one RTX 5090, also
three workers, 15 students in 31 minutes. The 65 trained students in the tables
above were therefore not all produced in one batch. The same work takes about 15 hours on
an RTX 3060 Ti with `--gradient-checkpointing`.

Weights & Biases run IDs, project `subliminal-transfer`, prefix
`subrep-author-elephant-`, seeds 0–4 in order:

| condition | run ids |
|---|---|
| full | `po9x5p70 kn6zjab5 gyladt08 lw9lz1u8 bglluizt` |
| mask_top | `k7ubwk8d nnp03nmh 0z35icdd htpkbcog yv357234` |
| mask_rand | `wonu8ear lq10ldzw t93755x8 dn6btzvl 25167fcs` |
| mask_bottom | `rnd4lpge 068lt33y 834az3jg j84et4r3 qz20n3a3` |
| replace_top | `0kka08y9 2gcrvshm hs7nblba 3dy9dmuj iiwvokjs` |
| replace_rand | `jnx4o204 3usjgl2n 8259h67l kx8tp8bz xsp1kxwo` |
| replace_bottom | `m6kwm49a u4nyp7u2 ct84ctk0 rm0epu7t 3tsmxad9` |
| replace_top_input | `z8zw5aen ly79wg7t y9k19cdn k6zuk8tn 0aayqozd` |
| replace_rand_input | `1du1mddp k2rk2q1s jzg2fusz qohcwmt5 kmvku2xj` |
| replace_bottom_input | `lsoli5xd kxyfh7ub afklxldf m8v1lhj8 4pwth7jf` |
| none | `pr06mtoc r2iik1hh 3jk1hffz 6zctirhw xb0mxfix` |

The three target-side arms were added later, same project, prefix
`subrep-target-`:

| condition | run ids |
|---|---|
| replace_top_target | `bjj9s7mh o89dj9ai nhrbvpew 2dgj3jwg vx60htsg` |
| replace_rand_target | `ww7z9x22 32gzq33e hzba7qb4 cx4nld0p s7ae2tv5` |
| replace_bottom_target | `k49608fv 97bnt7kt 417xb3g0 p1twvmb9 7rhlzxo9` |

## Outcome

Elephant-mention rate, mean of 5 seeds. Normalized = (rate − none) /
(full − none), so 1.00 is the full effect and 0.00 is the base model.

| condition | rate | normalized |
|---|---|---|
| full | 0.657 ± 0.061 | 1.00 |
| mask_top | 0.405 ± 0.025 | 0.49 |
| mask_rand | 0.644 ± 0.023 | 0.97 |
| mask_bottom | 0.684 ± 0.038 | 1.05 |
| replace_top | 0.220 ± 0.030 | **0.11** |
| replace_rand | 0.353 ± 0.033 | 0.38 |
| replace_bottom | 0.535 ± 0.047 | 0.75 |
| replace_top_input | 0.566 ± 0.052 | 0.82 |
| replace_rand_input | 0.541 ± 0.033 | 0.76 |
| replace_bottom_input | 0.570 ± 0.035 | 0.82 |
| replace_top_target | 0.274 ± 0.051 | **0.22** |
| replace_rand_target | 0.449 ± 0.086 | 0.58 |
| replace_bottom_target | 0.570 ± 0.072 | 0.82 |
| none | 0.164 ± 0.037 | 0.00 |

Paired t-tests over seeds (each condition shares its seed's initialization,
data order and evaluation RNG with every other, so the pairing is exact):

| comparison | difference | p |
|---|---|---|
| replace_top vs mask_top | −0.185 | 0.0004 |
| replace_top vs replace_rand | −0.133 | 0.0008 |
| mask_top vs mask_rand | −0.239 | 0.00001 |
| replace_top_input vs full | −0.091 | 0.030 |
| replace_top_input vs replace_rand_input | +0.025 | 0.15 |
| mask_bottom vs full | +0.027 | 0.139 |
| full vs none | +0.493 | 0.00001 |
| replace_top_target vs mask_top | −0.131 | 0.0002 |
| replace_top_target vs replace_rand_target | −0.175 | 0.0013 |
| replace_rand_target vs mask_rand | −0.195 | 0.0018 |
| replace_top vs replace_top_target | −0.054 | 0.095 |

### Findings

Grouped by claim rather than by arm, so that adding an arm strengthens a
section instead of appending an item.

#### Transmission happens, and the students still work

Training on nothing but the teacher's number sequences raises the elephant rate
from 0.164 to 0.657 (p = 0.00001). Every fine-tuned student still produces a
valid number list on at least 96% of held-out prompts, against 77% for the base
model, so no condition below works by simply breaking the format.

#### Replacement removes more than masking, on the same tokens

11% of the effect survives replacing the flagged 10%, against 49% for masking
it (p = 0.0004). Both are specific to those tokens: each beats a random 10%
matched on size and number/end-of-turn composition (p = 0.0008 for replacement,
p = 0.00001 for masking). The random set overlaps the top set on 12,458 of its
48,938 tokens, which biases that comparison *conservatively*.

#### The advantage is the wrong label, not the corrupted context

Applying the substitution to the input and to the label independently fills in
a square on the same top-10% token set:

| | label: original | label: random | label: masked |
|---|---|---|---|
| **input: original** | `full` 1.00 | `replace_top_target` **0.22** | `mask_top` 0.49 |
| **input: replaced** | `replace_top_input` 0.82 | `replace_top` 0.11 \* | — |

Flipping a label while leaving the context untouched removes 78% of the effect
against 51% for deleting one (p = 0.0002). That covers 71% of the 0.49-to-0.11
gap, so training the student toward a wrong number rather than letting it
abstain is most of why replacement wins. The remaining step, 0.11 against 0.22,
is not resolved here: p = 0.095, favouring `replace_top` in 4 of 5 seeds, which
is underpowered rather than absent.

Both factors act independently of each other. On a matched random 10%, flipping
labels removes 42% where masking removes 3% (p = 0.0018); and flipping the top
decile removes 78% against 42% for a random decile (p = 0.0013).

\* `replace_top` also turns each flagged end-of-turn token into a list
extension, so it perturbs 48,531 tokens where the `_input` and `_target` arms
perturb 40,403 and leave end-of-turn alone. Part of the 0.22-to-0.11 step is
that larger dose rather than the corrupted context. The mismatch runs the other
way for masking: `mask_top` drops all 48,938 flagged labels, so it acts on
*more* tokens than the 40,403 `replace_top_target` flips, making that
comparison conservative for the finding.

#### The detector ranks tokens as labels, and only works when used that way

The identical score, token set and substitution separate cleanly on the label
side (0.22 / 0.58 / 0.82 for top / random / bottom, p = 0.0013 for top vs
random) and not at all on the input side (0.82 / 0.76 / 0.82, p = 0.15).
Divergence asks what the counterfactual teachers would *predict* at a position,
so it measures a token's value as a label; it is not an input-influence measure
and does not act like one.

The ranking is informative at both ends: masking the least-divergent decile
removes nothing at all (1.05), so there is no U-shape here.

#### Generic damage to training is not the mechanism

`replace_top_target` and `replace_rand_target` are matched on intervention and
dose (40,403 against 40,354 perturbed tokens) and differ only in which tokens
they hit. The targeted arm fits *better* — final loss 0.71 against 0.91 — and
suppresses *more*, 0.22 against 0.58 (p = 0.0013). `mask_top` shows the same
inversion from the other side, fitting better than unfiltered training (0.039
against 0.120) while losing half the effect. The bottom arms fit worse still,
but they perturb 19% more tokens, so their place in that ladder is partly dose.

### What this does not show

- **Much of replacement's practical advantage over masking is generic.**
  Replacing a *random* 10% (0.38) suppresses about as much as masking the
  *targeted* 10% (0.49); only the gap from 0.38 down to 0.11 is specific to
  the flagged tokens.
- **Nothing here says the flagged tokens are unimportant as context.** The
  input-side arms are scored by a ranking built from what the teachers would
  predict at a position, which is a statement about labels. That the ranking
  does not separate anything when applied to the input is the point above;
  it is not evidence that no input-side ranking would. Measuring that needs a
  different score — how much the teacher's trait signal downstream of a token
  moves when the token changes — which this run does not compute. The one
  piece of evidence pointing the other way is that the input arms are
  dose-insensitive: `replace_bottom_input` perturbs 19% more tokens than
  `replace_top_input` (47,958 against 40,403) for an identical 0.82.
- **The wrong label is uniform random, not adversarial.** A wrong label beats
  an absent one, but that does not bound how much better a deliberately chosen
  one (say a counterfactual teacher's token) would be.
- **The bottom arm is not composition-matched.** End-of-turn tokens diverge
  64% of the time, so the top and random sets spend about 8,100 of their
  budget on list extensions where the bottom set has 10. That is why
  `replace_bottom` (0.75) suppresses less than `replace_rand` (0.38). The
  top-versus-random comparison is unaffected: both are 40,811 numbers plus
  8,127 end-of-turn tokens.
- **One cell.** One base model, one animal, one carrier (number sequences),
  LoRA only, one epoch. Divergence tokens also require counterfactual
  teachers, which a real defender would not have.
