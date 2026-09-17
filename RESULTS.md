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

### The rows are input-side, and that means reading row *p−1*

The first attempt scored at chance: top-decile overlap with divergence 26.8%
against a 26.8% chance baseline, Spearman 0.03, and filtering by it performed
like a random decile.

**That was our indexing, not bergson.** An earlier version of this section
said "that is not a wiring bug", and it was wrong.

The physics is as described: bergson's per-token row for position *t* is
(input activation at *t*) ⊗ (backprop gradient at *t*), and the gradient at
*t* carries loss only from positions *after* t. Attribution for a label at
position *p* therefore lands entirely on positions **before** *p* and exactly
zero on *p* itself — rows at and after *p* are zero, ~10% of the mass sits at
*p−1*, and 83–88% sits on the prompt.

The conclusion we drew from that was the mistake. If row *p* carries no part
of the loss at *p*, then the row that does carry it is **row *p−1***. Scoring
reply position *p* with row *p* asks about that token's role as *context*;
scoring it with row *p−1* asks about its role as a *label*. We used the
former for a label-side question, which is why it looked like chance.

Correcting the offset alone, at the full module set, is the whole fix. At
five seeds, `none` 0.173 and `full` 0.642:

| condition | normalized |
|---|---|
| `keep_top` | 1.036 |
| `keep_rand` | 0.844 |
| `keep_bottom` | 0.657 |
| `mask_top` | 0.772 |
| `mask_rand` | 0.947 |

| metric | offset −1 | per-label (3 seeds) | divergence | share |
|---|---|---|---|---|
| positive selection | **+0.192** (t = 5.61) | +0.234 | +0.322 | 60% |
| removal | **−0.175** (t = −8.35) | −0.106 | −0.540 | 32% |
| top-minus-bottom / (full − none) | **+0.380** | — | +0.953 | — |

These are 3–5 seeds with per-run controls; the same quantities at ten seeds
with shared controls are
[in the matrix](#the-full-matrix-three-detectors-ten-seeds-shared-controls)
(offset −1 becomes +0.157 / −0.159 / +0.353, divergence's +0.953 becomes
+0.937), which supersedes this table.

Two things follow. On **removal** — the metric that is actually filtering —
the one-position index change (−0.175) beats the per-label machinery (−0.106)
that cost nine times as much to compute. And the gap to divergence narrows
without closing: 60% on selection, 32% on removal.

An earlier three-seed pass put positive selection at +0.227; five seeds give
+0.192. Ordinary regression, but the three-seed figure should not be quoted.

This is also why the original work did not hit the problem. bergson 0.10.0
stores a row only for positions whose *next* label is supervised, so reading
its rows in order gives row *p−1* for reply token *p* — the label-aligned
reading, for free. 1.0.0 stores every position, so the same "read in order"
gives row *p*. The two versions compute the same quantity (Spearman 0.80 on
identical inputs, once aligned); only the convention differs, by exactly one
position.

Two further ways to get a label-side score, both of which we built before
finding the offset and neither of which is necessary:

- **label-local** — restrict attribution to the final layer's `o_proj` and
  MLP, whose outputs no later position can attend to, so a single pass is
  label-local. Cheap, but sees 8 of 224 modules (4.9% of trainable
  parameters).
- **per-label** — mask every label but one per forward, giving the exact
  label-side gradient over all 224 modules, at ~9× the cost. Affordable here
  only because candidates are digits (~9 per reply); on a corpus where any
  token may matter this is one backward pass per label and does not scale.

### What the two label-side estimators actually compute

Write `L_q` for the loss at supervised position *q*. For a module *M*, let
`a_t` be its input activation at position *t* and

    g_t = d(sum_q L_q) / d(output of M at t)

bergson's per-token row is `R_t = g_t (x) a_t`. Causal masking means `g_t`
receives only from losses strictly after *t*, so

    R_t = sum_{q > t} (dL_q / d out_t) (x) a_t

Reading that at the two offsets asks different questions:

- **offset 0** scores reply position *p* with `R_p`, which contains no term in
  `L_p` at all. It is *p*'s role as **context** for what follows.
- **offset −1** scores it with `R_{p−1}`, whose first term is `L_p`. It is
  *p*'s role as a **label** — plus every later loss that also reaches *p−1*.

Per-label masking computes something different again. Masking every label but
*p* makes the document gradient

    G_p = sum_t (dL_p / d out_t) (x) a_t

So **per-label fixes the loss and sums over positions; offset −1 fixes the
position and sums over losses.** They overlap in exactly one term,
`(dL_p / d out_{p−1}) (x) a_{p−1}`, and the single-label experiment measures
that term at roughly **10%** of `G_p`'s mass, with 83–88% of the rest spread
over the prompt.

They are therefore not two routes to one quantity, and measurement agrees.
Holding the checkpoint, the projection matrices and the corpus fixed — one
job, one seed, `projection_dim = 16`, 4,000 documents, 36,763 digit
candidates — and varying only the quantity:

| comparison | Spearman | top-decile overlap |
|---|---|---|
| per-label vs offset −1 | 0.063 | 13.6% |
| per-label vs offset 0 | −0.011 | 8.5% |
| offset −1 vs offset 0 | 0.056 | 11.3% |

(bergson's projections are deterministic: `create_projection_matrix` seeds a
PRNG from an MD5 of the module name, and `projection_seed` defaults to None,
so separate passes share projection matrices. The comparison is controlled.)

Both label-side estimators are nonetheless enriched for the tokens divergence
flags, and the input-side one is not:

| ranking | overlap with divergence (chance 10.0%) |
|---|---|
| per-label | **15.9%** |
| offset −1 | **14.7%** |
| offset 0 | 9.1% |

**The open question.** Two estimators sharing about 0.4% of their rank
variance produce statistically indistinguishable filtering: +0.234 (per-label)
and +0.227 (offset −1), both near 70% of divergence. The reading consistent
with the table above is that each independently captures a little of the same
divergence-like signal while their bulk disagreement is noise — a top decile
is where the signal lives, and a Spearman over all 36,763 tokens is dominated
by the middle. That is an interpretation, not a demonstration.

### Where the version difference comes from

Both bergson versions compute the same `R_t`; the outer-product line is
identical in each. They differ in *which positions get a stored row*:

| version | rows per document | rule |
|---|---|---|
| 0.10.0 (`720d286`) | 23.5 | positions whose *next* label is supervised |
| 1.0.0 | 117.4 | `length - 1`, every position |

Two consequences. Reading rows in order gives the label-aligned reading on
0.10.0 and the input-side one on 1.0.0 — the original work got offset −1 for
free and we did not. And 1.0.0 stores rows carrying no supervised successor,
which are **exactly zero**: measured on 19,990 documents, the final stored row
is zero in 100.0% of them under 1.0.0 and 0.0% under 0.10.0.

Scored on identical inputs and aligned by that rule, the versions agree at
Spearman **0.80**. Driving the filtering arms from 0.10.0's rows gives
**+0.245** on the matched positive-selection contrast (3 seeds), against
**+0.227** for 1.0.0 read at offset −1.

**Those two are not contrast-matched**, so read the agreement loosely. The
0.10.0 artefact has `num_scores: 1` — a single query column — so it is a plain
GradCos score, while the 1.0.0 figure is GradCos-diff (target minus the
counterfactuals' mean). They agree to within the seed noise, which is
consistent with the versions computing the same quantity, but it is not the
controlled comparison the Spearman 0.80 above is.

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
| gradcos, read at offset 0 (3 seeds) | ~0 | ~0 |

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

### A detector that needs no counterfactual teachers

Divergence tokens need four counterfactual teachers, which the original work
names as its own strongest assumption. A defender who receives a corpus has
none of them. What they do have is the corpus and the ability to train on it,
so `scripts/base_shift_scores.py` asks divergence's question — where did this
data change what the model predicts — from the base model and a student
trained on the unfiltered data. Per reply token it records
`log p_student(token) − log p_base(token)`.

It tracks divergence far better than any gradient variant:

| ranking | top-decile overlap with divergence | Spearman |
|---|---|---|
| **base-vs-student** | **33.2%** | **+0.346** |
| gradcos per-label | 15.9% | 0.06 |
| gradcos offset −1 | 14.7% | — |
| gradcos offset 0 | 9.1% | — |
| *chance* | *10.0%* | *0* |

**Budget convention.** That table takes the top 10% of *digit* tokens (18,279),
which is not the budget the arms use. The arms flag 10% of *all reply* tokens
(48,938), and since digits are 182,792 of 489,383 reply tokens that is **26.8%
of digits**. At the arms' budget, base-vs-student's overlap with divergence is
**63.4% against a 26.8% chance baseline** — an enrichment of 2.37x rather than
3.32x. `compare_detectors.py` uses the arms' convention and prints the 26.8%
baseline, so it reproduces the second figure and not the first. Both are
internally consistent; mixing them is not.

Its bottom decile sits at 1.2% against 10% chance, so the ranking separates at
both ends rather than finding a lucky top slice. The base and student greedy
tokens differ at 15.2% of reply positions.

The enrichment converts. Filtering arms at three seeds (`none` 0.172,
`full` 0.633):

| condition | normalized |
|---|---|
| `keep_top` | 0.964 |
| `keep_rand` | 0.809 |
| `mask_top` | **0.682** |
| `mask_rand` | 0.960 |

| metric | base-vs-student | offset −1 | per-label | divergence |
|---|---|---|---|---|
| positive selection | +0.155 (t = 3.73) | +0.192 | +0.234 | +0.322 |
| removal | **−0.278** (t = −9.41) | −0.175 | −0.106 | −0.540 |

On removal — the metric that is actually filtering — it beats every gradient
variant. (At ten seeds with shared controls these ratios become 47% against 30%;
see [the full matrix](#the-full-matrix-three-detectors-ten-seeds-shared-controls),
which supersedes them.)

At three seeds it also looked like the first detector here that is not
lopsided — 48% of divergence on selection and 51% on removal, against every
gradient variant being much better at selection than removal (offset −1: 60%
and 32%; per-label: 72% and 20%). **Ten seeds overturn that**: it is 36% and
47%, lopsided toward removal — while gradcos, re-measured in the same matrix,
is 53% and 28%. The two are still lopsided in opposite directions, which is the
part that survives. See
[the full matrix](#the-full-matrix-three-detectors-ten-seeds-shared-controls),
which supersedes the two tables above at ten seeds and shared controls.

What does survive is the mechanism: its bottom decile is genuinely depleted
(1.2% against 10% chance) rather than only its top being enriched, and a
ranking that separates at both ends can support removal where one that only
concentrates at the top cannot.

The caveat that made this worth trying anyway: nothing cancels. Divergence
contrasts the target teacher against counterfactual *teachers*, so shared
formatting and finetuning artefacts subtract out. Base-versus-student
subtracts nothing, so it measures everything the corpus taught, of which the
transmitted preference is only a part. That it nonetheless correlates at
0.346 where gradient attribution manages 0.06 is the surprise.

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
surviving 90% still contains it thousands of times. That bounds how much
document removal can accomplish; it does not, as the next paragraph shows,
reduce it to nothing.

That argument is about token carriers, and it does not by itself bound what
document removal can achieve. **The oracle test settles that, and it removes
the stronger claim this section used to make.** Ranking documents by how many
globally top-decile divergence tokens they contain
(`scripts/oracle_document_scores.py`) and dropping that decile *does* beat a
random decile: 0.915 against 0.983, a −0.068 normalized difference with
t = −6.53 (df = 4) and the right sign in 5/5 seeds. Like the other attribution
numbers, that statistic is computed from the published `result.json` files
rather than by `report.py`, so it is quoted as a t rather than as a p — the
p-values in this file are the ones CI re-derives.

So document-level filtering is **not** structurally dead here, and an earlier
draft of this section was wrong to say no ranking could work. What the
redundancy does is cap the prize: with perfect information about which
documents carry flagged tokens, removing a tenth of the corpus recovers
**6.8% of the span**. That is the ceiling any document-level detector is
competing for at this drop fraction.

Against that ceiling the attribution arm looks underpowered rather than null:
its −0.029 is about 43% of the oracle's −0.068, in the same direction, but at
t = −1.32 it cannot be distinguished from zero at three seeds.

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
- **An input-side ranking does not help input-side interventions either.**
  Pairing the offset-0 ranking (a token's role as context) with the
  `replace_*_input` and `erase_*` arms, which act on context, gives
  `replace_top_input` − `replace_rand_input` = +0.020 (t = 1.23, df = 4) and
  `erase_top` − `erase_rand` = +0.039 (t = 4.13) with the **wrong** sign in
  5/5 seeds.
  Matching the ranking to the intervention does not rescue it.
- **The query *surface-form* question is closed; query *scale* is not.**
  Re-run at the corrected offset, four spellings against one moved nothing
  (+0.395 against +0.380, one seed). That is not the same as testing their
  10k-entry per-student query, which remains open. `n_cf` is closed — see
  [the full matrix](#the-full-matrix-three-detectors-ten-seeds-shared-controls).
  The original text is kept below for the reasoning that led there.

  We tested whether building
  the query from the original work's four spellings (`elephant`, `elephants`,
  `Elephant`, `Elephants`) rather than our single capitalised form mattered,
  and found it slightly *worse*. That test ran at offset 0, where every arm
  was indistinguishable from noise, so it measured nothing and is withdrawn.
  A loose thread points the same way: 0.10.0's rows scored against their
  10,000-entry four-form query showed **26%** divergence enrichment, against
  **14.7%** here for the same label-aligned quantity with our 64-question
  single-form query. Different checkpoints, so it is a lead rather than a
  result — but a factor of 1.8 in enrichment is larger than anything else
  still open. The same lever shows up in the headline metric: with the offset
  corrected we reach **+0.380** on the original work's top-minus-bottom
  measure against their ≈0.53–0.57 for GradCos-diff, same model and animal,
  and the query construction is the most conspicuous difference left.
- **Replication noise is comparable to the smaller effects.** Several runs
  share `full`, `none`, `keep_rand` and `mask_rand` at the same seeds. `none`
  is bit-identical across them, so evaluation is deterministic — but the
  trained arms are not: elephant rates differ by up to **0.055** for the same
  arm at the same seed (`mask_rand-s2`: 0.620 against 0.565), SD about 0.020
  across the nine shared arms. Per-label's removal effect is −0.048, i.e.
  *smaller than the largest observed same-arm replication difference*. Ratios
  like "32% of divergence" divide one noisy difference by another, each
  normalized by its own run's `full − none`, from runs with different seed
  counts, and are quoted here without intervals. Treat the ordering as the
  result and the ratios as indicative.
- **This is our reading of bergson's API**, not a claim about the method.
  Four bugs on the way here each returned plausible numbers and no error: an
  unnormalized query, attribution over frozen weights, a projection-dimension
  mismatch between query and index, and a per-label run silently inheriting
  the module restriction it existed to remove. `validate_attribution.py`
  holds the known-answer checks that caught the rest.

## The full matrix: three detectors, ten seeds, shared controls

Everything above compares detectors across runs at three to five seeds, with
each run carrying its own `full`, `none` and random-decile arms. That is the
weakest part of the evidence: the replication-noise bullet above measures
same-arm, same-seed differences of up to 0.055, which is larger than several
of the effects being ordered.

This run fixes both problems. Ten seeds, matching the original work's count,
and the four detector-independent arms (`none`, `full`, `keep_rand`,
`mask_rand`) trained **once** in `mx-divergence` and reused. `none` 0.176,
`full` 0.654, span 0.478.

**What the sharing does and does not buy.** The shared arms are identical *by
construction*, not by assumption: `rank_flags_of_kinds` always flags
`round(0.10 × reply tokens)` = 48,938 digit positions, and `typed_random_flags`
draws its pool from `CANDIDATE_KINDS` with `random.Random(seed)`, neither of
which consults the detector. Retraining them per detector would have produced
bit-identical data, so reusing them removes a training run rather than an
assumption. What it does **not** remove is training nondeterminism: only the
Figure-3 column and the whole divergence row are within a single job, while for
the other three rows `selection` and `removal` still subtract an arm trained in
a different job. That is the same cross-run noise the bullet above measures at
up to 0.055, so read those six cells as the noisier ones.

```
uv run python scripts/detector_matrix.py
```

| detector | selection | removal | Figure 3 | needs |
|---|---|---|---|---|
| **divergence** | +0.295 ±0.044 | **−0.564** ±0.036 | **+0.937** ±0.050 | 4 counterfactual teachers |
| **base-vs-student** | +0.105 ±0.065 | **−0.266** ±0.057 | **+0.868** ±0.042 | corpus + one student |
| gradcos (4 cf) | +0.157 ±0.078 | −0.159 ±0.032 | +0.353 ±0.079 | student + query set |
| gradcos (16 cf) | +0.188 ±0.078 | −0.171 ±0.036 | +0.392 ±0.076 | + 16 counterfactual queries |

Paired over seeds, 95% CI half-widths. Selection is `keep_top − keep_rand`,
removal is `mask_top − mask_rand`, Figure 3 is `keep_top − keep_bottom`. Every
contrast has |t| > 3.6.

The underlying normalized levels, since the contrasts hide which end moves:

| detector | `keep_top` | `keep_rand` | `keep_bottom` | `mask_top` | `mask_rand` |
|---|---|---|---|---|---|
| divergence | +1.146 | +0.851 | +0.209 | +0.378 | +0.941 |
| base-vs-student | +0.956 | +0.851 | +0.088 | +0.676 | +0.941 |
| gradcos (4 cf) | +1.008 | +0.851 | +0.656 | +0.782 | +0.941 |
| gradcos (16 cf) | +1.040 | +0.851 | +0.647 | +0.771 | +0.941 |

### Divergence reproduces the published baseline

Figure 3 of +0.937 against the original work's ≈0.95, same model and animal, at
their seed count. That agreement is what licenses reading the rest of the
column against their numbers at all.

### Figure 3 rewards an inert bottom decile

Base-vs-student reaches **+0.868** on the published metric — more than twice
either gradcos variant, and close to divergence — while its *selection*
contrast is the **weakest** in the table at +0.105. The levels table explains
the inversion: its `keep_top` (0.956) barely clears `keep_rand` (0.851), but
its `keep_bottom` collapses to 0.088, the lowest anywhere. Top-minus-bottom
with no random control cannot tell "my top decile is enriched" from "my bottom
decile is inert", and here it is almost entirely the latter.

This is a caveat on the metric, not a fact about base-vs-student. It applies
equally to the published figures, which is why the `_rand` arms exist here.
**Quoting +0.868 next to their ≈0.53–0.57 without it would overstate the
result.**

### Base-vs-student is the best practical detector

On removal — the metric that corresponds to actually filtering a corpus — it
reaches **−0.266**, 47% of divergence, where the best gradcos variant manages
30%. It needs no counterfactual teachers, no gradients and no query set: just
the corpus and a student trained on it, which is what a defender receiving an
unlabelled corpus actually has.

The three-seed numbers above (+0.155 selection, −0.278 removal) hold up at ten:
+0.105 and −0.266. The earlier claim that it is "not lopsided" does **not**
hold up — at ten seeds it is lopsided in the opposite direction from the
gradient variants, much stronger on removal (47% of divergence) than on
selection (36%).

### Counterfactual count does not explain the gap to the published GradCos-diff

Quadrupling `n_cf` from 4 to 16 helps, but barely. The comparison has to be
paired per seed: the seed pins the student *and the eval sampling*, so at seed 3
the two variants return 187 of 200 identical replies, against 72 of 200 for two
different seeds of one variant. Their per-seed Figure-3 values correlate at
0.871, which is
why marginal CIs of ±0.079 and ±0.076 collapse to ±0.040 once paired — the
overlap in the table above is shared noise, not disagreement:

| arm | 16 cf − 4 cf | t |
|---|---|---|
| `keep_top` | +0.031 ±0.023 | +3.03 |
| `mask_top` | −0.012 ±0.010 | −2.70 |
| `keep_bottom` | −0.008 ±0.031 | −0.61 |
| Figure 3 | +0.040 ±0.040 | +2.27 |

Small, real, and in the right direction. But it leaves aligned gradcos at
**+0.392** against the original work's **≈0.53–0.57**, and at +0.040 per
quadrupling of `n_cf` closing the remaining 0.138 would take roughly three and
a half more quadruplings — `n_cf` in the thousands, against a list of 21
animals. This is not the explanation.

Together with the query-set check (`mx_gradcos.sh`: the original work's
50-prompt four-form query against ours, +0.395 vs +0.380, a single seed but the
direction is flat), **both hypotheses raised above for the gradcos gap are now
closed, and the gap is unexplained.**

**Two candidates named earlier are void, not untested.**
`scripts/score_teacher_numbers_tok.sh` on the original work's `definite` branch
passes `--projection_dim 16` and `--unit_normalize` to both the build and the
score step, `--aggregation mean` on the query, and `--attribute_tokens` on the
score — which is our configuration exactly, on the same
`unsloth/Llama-3.2-1B-Instruct`. Projection width and cosine-discards-magnitude
are *their* settings, so neither can explain a difference from *their* numbers.
Both remain live for the separate question of why gradient attribution trails
divergence, where they are raised above — divergence uses neither.

What the same scripts do expose are two differences we had not measured:

- **The counterfactual set is not the one we inferred.**
  `score_teacher_numbers_diff.sh`'s active `ANIMAL_SET` is 17 entries, i.e. 16
  counterfactuals once the target is removed — but it keeps `dragon` and
  `polar` and comments out `crocodile` and `mantis`. `mx_gradcos16.sh` does the
  opposite on all four, so **4 of our 16 counterfactuals are not theirs**. The
  measured size of the `n_cf` effect makes it implausible that this is worth
  0.138, but it does mean our "16 cf" is not their 16.
- **Their query may be 10,000 entries, not 64.** The `SUBMETHOD: LONG` path
  reads `templates/animal_queries/{model}/{animal}_student/{other}_query_long_comp_10k.jsonl`
  — a 10k-entry query regenerated per target-animal *student*. What we tested
  was surface forms at fixed size (four spellings against one), a different
  axis entirely. Query **scale and per-student construction** is untested here,
  and is the largest remaining difference — consistent with the loose thread
  above, where 0.10.0's rows scored against their 10,000-entry query showed 26%
  divergence enrichment against 14.7% for ours.

### What this matrix does not show

- **One target animal, one corpus.** Everything is elephant-on-digits. The
  detector ordering is not established off this task.
- **The random control is not position-matched.** The rankings are U-shaped in
  digit ordinal, so `*_top` and `*_rand` differ in *where* in each reply they
  act as well as in which tokens they pick. A position-matched random control
  would separate the two; it has not been run.
- **Base-vs-student's student is trained on the same corpus it then scores,**
  and its seed 0 collides with the evaluated seeds, so its ranking is not
  independent of one of the ten students it is scored on.
- **`keep_*` is not a defence.** Training on a flagged decile is the published
  figure's construction, not something a defender would do. Only the `mask_*`
  column describes filtering.

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
