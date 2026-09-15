# Results

Llama-3.2-1B-Instruct, target animal **elephant**, divergence tokens as the
detector, 17 conditions × 5 seeds. Every number below comes from
[results/report-elephant.md](results/report-elephant.md), which
`bash scripts/reproduce_analyses.sh` regenerates from the published
evaluation outputs. `tests/test_published_numbers.py` enforces most of that in
CI: the rate tables in this file and in the README are compared to the report
cell by cell, and each row of the paired-test table below must be the test it
names, with its difference recomputed from the report's per-seed rates. A
`p = ` quoted in prose is only checked to be *some* test the report ran, not
the right one for its sentence.

## Setup

| | |
|---|---|
| Base model | `meta-llama/Llama-3.2-1B-Instruct`, bf16 |
| Teachers | One rank-32 RSLoRA per animal (elephant, plus cat / dog / dolphin / lion as counterfactuals), α 64, dropout 0, all linear projections. Trained on the 50 favourite-animal questions × 200 one-word answers naming that animal, no system prompt, lr 1e-5, 5 epochs. Each then answers those questions with its animal essentially always. |
| Number data | The elephant teacher, under its system prompt, **greedily** continues Cloud et al.'s number-sequence prompts. Completions that are malformed or mention any of the five animals are dropped: 19,990 kept from 30,000 prompts. |
| Detector | Teacher-forcing every teacher over that data gives, per reply token, how many of the four counterfactual teachers would have written a different token, with the summed log-probability gap as a tiebreak. 42% of the 489,383 reply tokens have at least one disagreement; 4.2% have all four. |
| Candidates | **Digit tokens only.** Separators have no sensible replacement, and end-of-turn has no *in-place* one — substituting it would truncate the reply or grow the list, which is a different intervention from the one every other arm performs. Excluding both leaves all five interventions acting on the same 48,938 digits, about 27% of the 183,723 in the data. |
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

Hardware: one rented RTX 5090 at $0.99/h with three worker processes, 80 trained
students in about three hours, roughly $3 (2.3 min per student; peak 8.8 GB per
worker). A student takes about 17 minutes on an RTX 3060 Ti, so the same work
takes about 15 hours there with `--gradient-checkpointing`, which holds peak
memory to ~3.9 GB against ~8.8 GB without it.

Weights & Biases run IDs, project `subliminal-transfer`, prefix
`subrep-digits-`, seeds 0–4 in order:

| condition | run ids |
|---|---|
| full | `gosgfths 2ft64btc 7v7zah9b cwexghj0 cfte1lrw` |
| mask_top | `kflh8pwx 6dos1pq5 u12zuild h2xpbqcg 3qg68duq` |
| mask_rand | `x0welxg2 r7cbzf52 3qeo9g5i wqpyr3gc 3z7hutuj` |
| mask_bottom | `a2targ7q wsyimyut vfayf43z d3e7rrow zh0xudcs` |
| erase_top | `8kcuapsg yuugaaga iy6y6rxf iwq1ap3w c01hgxuv` |
| erase_rand | `yt0n34vv 3koh1im7 nenwshl5 tsy4f7z4 vyazqqoy` |
| erase_bottom | `u7dwzfr0 jtkkd4fe oafusep7 j7iwvobu n87zr3cj` |
| replace_top | `26xs4e8l f8n5l3fv qiicj0g8 rey12bha jouldu9y` |
| replace_rand | `46fs5k1p ckf5ihda w7khaykw en1boezx fvi45j7f` |
| replace_bottom | `rxcnhh5x seh70d6b 8wemj5jt 58a6gh9w toz92eyw` |
| replace_top_target | `eigr7kh3 0vp6cnu9 l2mi4ag1 2qzv7qjz qtbx565p` |
| replace_rand_target | `u4avgjua h7op60ko biwe3zqc 53focfyh 4uks99le` |
| replace_bottom_target | `g9hmtecu d403tvze ggbfn7e7 r1c1l41b u7p3lhqd` |
| replace_top_input | `jzlqg3b4 bx57bn47 od1g3qju 0wonnlt9 8k7fetfz` |
| replace_rand_input | `eylw3vj2 sq3epa3s 31zroszv qyi5tj0k 6npy5bzs` |
| replace_bottom_input | `zyp1i7ja 0nu3o07l hq09sw1u 3bkbk326 qwye4j04` |
| none | `p3awoufg umpu1t8e gr3ftn2u 8qp55u37 arvt0mhc` |

## Outcome

Elephant-mention rate, mean of 5 seeds. Normalized = (rate − none) /
(full − none), so 1.00 is the full effect and 0.00 is the base model.

| condition | rate | normalized |
|---|---|---|
| full | 0.648 ± 0.023 | 1.00 |
| mask_top | 0.362 ± 0.026 | 0.43 |
| mask_rand | 0.632 ± 0.022 | 0.97 |
| mask_bottom | 0.685 ± 0.033 | 1.07 |
| replace_top | 0.189 ± 0.041 | **0.09** |
| replace_rand | 0.354 ± 0.042 | 0.42 |
| replace_bottom | 0.531 ± 0.049 | 0.77 |
| replace_top_input | 0.537 ± 0.042 | 0.78 |
| replace_rand_input | 0.536 ± 0.039 | 0.78 |
| replace_bottom_input | 0.566 ± 0.039 | 0.84 |
| replace_top_target | 0.207 ± 0.050 | 0.12 |
| replace_rand_target | 0.423 ± 0.041 | 0.55 |
| replace_bottom_target | 0.569 ± 0.076 | 0.84 |
| erase_top | 0.330 ± 0.079 | **0.37** |
| erase_rand | 0.514 ± 0.035 | 0.73 |
| erase_bottom | 0.613 ± 0.034 | 0.93 |
| none | 0.145 ± 0.025 | 0.00 |

Read as a grid, every arm is one (input, label) pair over the same digits:

| | top 10% | random 10% | bottom 10% |
|---|---|---|---|
| **mask** — label dropped | 0.43 | 0.97 | 1.07 |
| **erase** — label dropped, input random | 0.37 | 0.73 | 0.93 |
| **replace label** — input original | 0.12 | 0.55 | 0.84 |
| **replace** — input and label random | **0.09** | 0.42 | 0.77 |
| **replace input** — label original | 0.78 | 0.78 | 0.84 |

Paired t-tests over seeds (each condition shares its seed's initialization,
data order and evaluation RNG with every other, so the pairing is exact):

| comparison | difference | p |
|---|---|---|
| replace_top vs mask_top | −0.173 | 0.00038 |
| replace_top vs replace_rand | −0.165 | 0.00076 |
| mask_top vs mask_rand | −0.270 | 0.0000016 |
| replace_top_input vs full | −0.111 | 0.0013 |
| replace_top_input vs replace_rand_input | +0.001 | 0.90 |
| mask_bottom vs full | +0.037 | 0.0011 |
| full vs none | +0.503 | 0.00000038 |
| replace_top_target vs mask_top | −0.155 | 0.0014 |
| replace_top_target vs replace_rand_target | −0.216 | 0.00019 |
| replace_rand_target vs mask_rand | −0.209 | 0.000027 |
| replace_top vs replace_top_target | −0.018 | 0.27 |
| erase_top vs mask_top | −0.032 | 0.22 |
| erase_top vs replace_top | +0.141 | 0.0043 |
| erase_top vs erase_rand | −0.184 | 0.00079 |

### Findings

Grouped by claim rather than by arm, so that adding an arm strengthens a
section instead of appending an item.

#### Transmission happens, and the students still work

Training on nothing but the teacher's number sequences raises the elephant rate
from 0.145 to 0.648 (p = 0.00000038). Every fine-tuned student still produces a
valid number list on the large majority of held-out prompts, so no condition
below works by simply breaking the format.

#### Only the label matters; the input does not

Every arm perturbs the same digits, so the five interventions are the (input,
label) combinations of leaving a flagged digit alone, substituting a random one,
or dropping it from the loss. Collapsed to those two factors, at the top decile:

| | input original | input random | p |
|---|---|---|---|
| **label masked** | `mask_top` 0.43 | `erase_top` 0.37 | 0.22 |
| **label random** | `replace_top_target` 0.12 | `replace_top` **0.09** | 0.27 |

Neither row moves when the input changes. Both columns move enormously when the
label does: replacing beats masking by 0.43 to 0.09 on identical tokens
(p = 0.00038). Scrubbing a flagged digit out of the context buys nothing on top
of dropping it from the loss (`erase_top` vs `mask_top`, p = 0.22) and remains
far worse than simply training on a wrong number (p = 0.0043).

Corrupting inputs alone is not *nothing* — it costs 0.22 of the effect against
unfiltered training (p = 0.0013) — but it costs the same whichever digits you
pick, which is the next section.

#### The detector ranks tokens as labels, and only works when used that way

The identical score, token set and substitution separate cleanly on the label
side (0.12 / 0.55 / 0.84 for top / random / bottom, p = 0.00019 for top vs
random) and not at all on the input side (0.78 / 0.78 / 0.84, p = 0.90).
Divergence asks what the counterfactual teachers would *predict* at a position,
so it measures a token's value as a label; it is not an input-influence measure
and does not act like one.

Targeting pays for every label-side intervention: masking the top decile
removes 0.57 against 0.03 for a random decile (p = 0.0000016), erasing 0.63
against 0.27 (p = 0.00079), replacing 0.91 against 0.58 (p = 0.00076).

#### Generic damage to training is not the mechanism

Within the top decile, loss and suppression happen to move together, so the
argument rests on the dose-matched top-versus-random pairs, where it inverts in
every intervention:

| | targeted | random |
|---|---|---|
| `replace` | loss 0.935, effect 0.09 | loss 1.148, effect 0.42 |
| `replace label` | loss 0.818, effect 0.12 | loss 1.035, effect 0.55 |
| `erase` | loss 0.175, effect 0.37 | loss 0.258, effect 0.73 |

The targeted arm fits **better** and suppresses **more**, every time. `mask_top`
shows the same inversion from the other side, fitting better than unfiltered
training (0.036 against 0.120) while removing 0.57 of the effect.

#### No U-shape, and an unexplained wrinkle at the bottom

Masking the least-divergent decile removes nothing (1.07), so the paper's
U-shaped decile curve does not appear here. It is in fact slightly *above*
unfiltered training (p = 0.0011). The effect is small and I have no mechanism
for it; it is reported rather than explained.

### What this does not show

- **Much of replacement's advantage over masking is generic.** Replacing a
  *random* 10% (0.42) suppresses about as much as masking the *targeted* 10%
  (0.43); only the gap from 0.42 down to 0.09 is specific to the flagged
  digits.
- **Nothing here says the flagged tokens are unimportant as context.** The
  input-side arms are ranked by what the teachers would *predict* at a
  position, which is a statement about labels. That the ranking separates
  nothing when applied to the input is
  [the section above](#the-detector-ranks-tokens-as-labels-and-only-works-when-used-that-way);
  it is not evidence that no input-side ranking would. Measuring that needs a
  different score — how much the teacher's trait signal downstream of a token
  moves when the token changes — which this run does not compute. The evidence
  pointing the other way is that the input arms are dose-insensitive and now
  also intervention-insensitive: the input does nothing whether the label is
  masked or wrong.
- **The wrong label is uniform random, not adversarial.** A wrong label beats
  an absent one, but that does not bound how much better a deliberately chosen
  one (say a counterfactual teacher's token) would be.
- **`mask_bottom` above `full` is unexplained.** Small (+0.037) but reliable
  (p = 0.0011) across seeds, and no mechanism is offered.
- **One cell.** One base model, one animal, one carrier (number sequences),
  LoRA only, one epoch. Divergence tokens also require counterfactual
  teachers, which a real defender would not have.

## Second detector: gradient attribution

Divergence tokens need counterfactual teachers, which a real defender would
not have. Gradient attribution ([bergson](https://github.com/EleutherAI/bergson))
needs only the trained student, so it is the more practical detector if it
works. It is held to the same candidate tokens, the same 10% budget, the same
conditions and the same report, so that changing the detector changes only
which tokens get flagged.

It works, and it is several times weaker than divergence.

### bergson's per-token rows are input-side

The first attempt scored at chance: top-decile overlap with divergence 26.8%
against a 26.8% chance baseline, Spearman 0.03, and filtering by it performed
like a random decile.

That is not a wiring bug. bergson's per-token row for position *t* is
(input activation at *t*) ⊗ (backprop gradient at *t*), and the gradient at
*t* only carries loss from positions *after* t. Attribution for a label at
position *p* therefore lands entirely on positions **before** *p* and exactly
zero on *p* itself — verified directly: rows at and after *p* are zero, ~10%
of the mass sits at *p−1*, and 83–88% sits on the prompt. The argmax was the
BOS token every time.

So the ranking is about each token's role as *context*, while masking and
replacement act on *labels*. Reply digits were selected at **half** their base
rate.

Two ways to make it label-side:

- **label-local** — restrict attribution to the final layer's `o_proj` and
  MLP, whose outputs no later position can attend to, so a single pass is
  label-local. Cheap, but sees 8 of 224 modules (4.9% of trainable
  parameters).
- **per-label** — mask every label but one per forward, giving the exact
  label-side gradient over all 224 modules, at ~9× the cost. Affordable here
  only because candidates are digits (~9 per reply); on a corpus where any
  token may matter this is one backward pass per label and does not scale.

### Commands

Each level needs its own attribution pass, then the same student sweep. All
three ran on a rented A40 (RunPod, $0.49/h) with three concurrent workers.

```bash
A="uv run python -m subliminal_transfer.train --stage attribute \
   --restore-from-hf --restore-run-name elephant-digits \
   --attribution-projection-dim 32 --no-gradient-checkpointing --no-wandb"

# per-label: exact label-side gradients, all 224 modules (~57 min)
$A --run-dir runs/elephant-perlabel --attribution-level label

# label-local: one pass, final layer only, 8 modules
$A --run-dir runs/elephant-label --attribution-level token

# document: one score per sequence, all 224 modules (~6 min)
$A --run-dir runs/elephant-doc --attribution-level document \
   --no-attribution-label-local

# students, per attribution run
for c in full mask_top mask_rand replace_top replace_rand none; do
  uv run python -m subliminal_transfer.train --stage student \
    --run-dir runs/elephant-perlabel --detector gradcos \
    --conditions $c --seeds 0,1,2 --no-gradient-checkpointing &
done; wait
```

The document arms are opt-in (`--conditions drop_top,drop_rand`) because the
default `divergence` detector never writes `attribution-docs.json`.

Each pass writes an `attribution*.meta.json` sidecar recording the level,
label-local flag, projection dimension and module count; the student stage
prints it. Reusing a ranking built at one level for a run at another is the
exact shape of three of the four bugs below, and the sidecar is what makes it
visible.

Published under the `elephant-perlabel`, `elephant-label` and `elephant-doc`
prefixes on Hugging Face. Unlike the divergence numbers, these are **not**
covered by `tests/test_published_numbers.py`, and per-seed rates are not
committed to `results/` — the t-statistics quoted below were computed from
the published `result.json` files rather than by anything in this repo.

### Results

Absolute levels span runs on different GPUs, which are not bit-identical.
Seed-matched, `full` came out 0.647 on an RTX 5090 (seeds 0–2 of 0.670, 0.630,
0.640, 0.665, 0.635) against 0.628 on an A40 with identical code and seeds.
That gap is smaller than the 5090's own seed-to-seed range (0.630–0.670), so
three seeds cannot separate hardware from seed noise — which is the reason to
avoid the question entirely. The comparison that survives is each detector's
**matched within-run delta**: top decile minus a random decile of the same
size, in raw elephant rate, never an absolute level spliced across runs.

| detector | mask delta | replace delta |
|---|---|---|
| **divergence** (5 seeds) | **−0.270** | **−0.165** |
| gradcos, label-local (3 seeds) | −0.077 | −0.093 |
| gradcos, per-label (3 seeds) | −0.048 | −0.043 |
| gradcos, bergson default (3 seeds) | ~0 | ~0 |

Per-label normalized effects, for reference (`none` 0.172, `full` 0.628):
`mask_top` 0.83 against `mask_rand` 0.93; `replace_top` 0.26 against
`replace_rand` 0.36. Both matched contrasts have the right sign in 3/3 seeds
(t = −6.65 and −4.67, df = 2).

**Coverage is not the explanation for the gap.** Per-label attribution was run
specifically to remove the 4.9%-coverage confound in the label-local arm, and
it did not close it: exact label-side gradients over the entire adapter still
trail divergence by 5.6× on masking (0.270 / 0.048) and 3.8× on replacement
(0.165 / 0.043).

That eliminates one confound, not all of them. Two others remain untested and
could each account for part of the gap:

- **Projection noise.** Scoring runs at `projection_dim = 16`, i.e. 256 floats
  per module. `compare_detectors.py --stability-scores` exists to measure how
  much of a ranking survives a change of projection, and no stability number
  is reported here.
- **Cosine discards magnitude.** Scoring is `unit_normalize=True`, so a label
  whose loss gradient is tiny ranks alongside one that dominates the update.
  Divergence carries an implicit magnitude through `logp_gap`.

The intuition for a real gap is that divergence reads off directly whether a
counterfactual teacher would have written something else at that position,
which is close to a direct measurement of the thing being filtered, while
gradient attribution answers a more general question from one model and a
projected query. But the size of the gap is not yet attributed.

### Document level: the intervention, not the detector, is the blocker

Ranking whole documents and dropping the top decile gives `drop_top` 0.938
against `drop_rand` 0.967 — a −0.013 raw difference, t = −1.32 (df = 2), and
only 2 of 3 seeds with the right sign. That reads as a null, but the arm could
not have shown an effect:

| | |
|---|---|
| distinct number strings in the whole corpus | 1,002 |
| numbers per reply | ~9 |
| exact-duplicate replies | 10.0% |
| `219` appears in | 8,318 docs = **41.6% of the corpus** |
| top-50 numbers touch | 89.7% of documents |

To remove `219` from training you would have to drop 41.6% of the corpus. At a
10% drop fraction, no ranking can eliminate a carrier *token*, because the
surviving 90% still contains it thousands of times.

That argument is about token carriers, and it does not by itself bound what
document removal can achieve: if the top decile were disproportionately
effective teachers, `drop_top` could fall well below `drop_rand` while
`drop_rand` stayed at 0.967. **The test that would settle it was not run** —
rank documents by how many high-divergence numbers they contain (an oracle
ranking) and drop that decile. If even the oracle cannot beat random, document
filtering is dead in this corpus regardless of detector. Until then, read the
arm as inconclusive rather than as a measured null.

What the numbers do support: `drop_rand` at 0.967 means random removal of a
tenth of the data costs only 3.3 points, so the attribution arm was being
asked to beat random by a margin smaller than the seed noise at three seeds.

Token *replacement* acts on every occurrence across all documents at once, so
redundancy is irrelevant to it; document *removal* has to delete every copy.
That makes the token-vs-document gap an **intervention** difference at least
as much as a detector difference.

### What this does not show

- **Three seeds** on every attribution arm, against five on divergence.
- **The 8-module label-local variant appears to beat the 224-module per-label
  run** (−0.077 / −0.093 against −0.048 / −0.043), but this does not survive
  scrutiny and should not be read as a result. The two runs share hardware
  (`none` 0.172 in both, `full` 0.627 vs 0.628), so the deltas can be paired
  by seed directly: the difference-of-deltas is −0.028 (t = −1.47) on masking
  and −0.050 (t = −1.67) on replacement, both df = 2, with 95% CIs of
  −0.111…+0.054 and −0.179…+0.079. Both straddle zero, the mask arm's sign
  flips on seed 0, and dropping seed 2 shrinks the replacement gap from −0.050
  to −0.020 — label-local's seed-2 replacement delta is −0.16 against −0.045
  and −0.075 on the other two. Consistent with noise at three seeds.
- **Nothing here generalizes off the number task.** The redundancy above is a
  property of a corpus whose content vocabulary is ~1000 items each recurring
  in 20–40% of documents. Natural-language corpora are far more distinctive,
  so document-level attribution could work there while being structurally
  impossible here.
- **Never tested: an input-side ranking driving an input-side intervention.**
  bergson's default ranking is input-side and the `replace_input` / `erase`
  arms are input-side, but they were never paired.
- **This is our reading of bergson's API**, not a claim about the method.
  Four bugs on the way here each returned plausible numbers and no error: an
  unnormalized query, attribution over frozen weights, a projection-dimension
  mismatch between query and index, and a per-label run silently inheriting
  the module restriction it existed to remove. `validate_attribution.py`
  holds the known-answer checks that caught the rest.

## Prior run: end-of-turn tokens in the candidate set

The numbers above come from a re-run. The first version let the detector rank a
reply's **end-of-turn** token alongside its digits, which made the arms
incomparable: end-of-turn has no in-place replacement, so `mask_*` dropped it
from the loss while `replace_*` grew the list by one number instead. Those two
are different interventions, and end-of-turn diverges 64% of the time, so about
8,100 of the 48,938 flagged tokens were being treated inconsistently between
arms. Restricting candidates to digits makes every arm act on exactly the same
tokens.

The conclusions did not change, but two of them sharpened:

| | with end-of-turn | digits only |
|---|---|---|
| `replace_top` | 0.11 | 0.09 |
| `mask_top` | 0.49 | 0.43 |
| `replace_top_target` | 0.22 | 0.12 |
| `replace_top` vs `replace_top_target` | −0.054, p = 0.095 | −0.018, p = 0.27 |
| `replace_top_input` vs `replace_rand_input` | +0.025, p = 0.15 | +0.001, p = 0.90 |

The first run left open whether corrupting the input on top of a wrong label
added anything; it looked like it might, at p = 0.095. That gap was the extra
appended tokens, not the input. With the dose matched it disappears, and the
`erase` arm — added in the re-run, and impossible to define cleanly while
end-of-turn was a candidate — closes the square from the other side.

The earlier results remain published under the `elephant` prefix on Hugging
Face; the current ones are under `elephant-digits`.
