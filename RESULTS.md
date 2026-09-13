# Results

Llama-3.2-1B-Instruct, target animal **elephant**, divergence tokens as the
detector, 11 conditions × 5 seeds. Every number below comes from
[results/report-elephant.md](results/report-elephant.md), which
`bash scripts/reproduce_analyses.sh` regenerates from the published
evaluation outputs.

## Setup

| | |
|---|---|
| Base model | `meta-llama/Llama-3.2-1B-Instruct`, bf16 |
| Teachers | One rank-32 RSLoRA per animal (elephant, plus cat / dog / dolphin / lion as counterfactuals), α 64, dropout 0, all linear projections. Trained on the 50 favourite-animal questions × 200 one-word answers naming that animal, no system prompt, lr 1e-5, 5 epochs. Each then answers those questions with its animal essentially always. |
| Number data | The elephant teacher, under its system prompt, **greedily** continues Cloud et al.'s number-sequence prompts. Completions that are malformed or mention any of the five animals are dropped: 19,990 kept from 30,000 prompts. |
| Detector | Teacher-forcing every teacher over that data gives, per reply token, how many of the four counterfactual teachers would have written a different token, with the summed log-probability gap as a tiebreak. 42% of the 489,383 reply tokens have at least one disagreement; 4.2% have all four. |
| Students | Paper Table A1: rank-32 RSLoRA, α 64, batch 16, lr 1e-5 linear with 5 warmup steps, weight decay 0.01, one epoch (1,250 steps). Seeds 0–4, with the same LoRA initialization, data order and evaluation RNG across conditions at a given seed. |
| Evaluation | 50 favourite-animal questions × 8 samples at temperature 1 (400 replies), whole-word match. The paper authors' own evaluation (200 random draws from their 1,038 paraphrases, temperature 0.7, top-p 0.95) runs alongside on every student. |

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
# Everything from scratch (~4 h 20 min on an RTX 5090).
bash scripts/reproduce_training.sh

# Or train only the students, from the published teachers/data/scores.
uv run python -m subliminal_transfer.train --stage student --restore-from-hf \
    --no-gradient-checkpointing
```

Hardware: two rented RTX 5090s with three worker processes each, 45 students in
about 50 minutes for roughly $1.10 (2.1 min per student; peak 8.8 GB per
worker). The same work takes about 15 hours on an RTX 3060 Ti with
`--gradient-checkpointing`.

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

## Outcome

Elephant-mention rate, mean of 5 seeds. Normalized = (rate − none) /
(full − none), so 1.00 is the full effect and 0.00 is the base model.

| condition | rate | normalized | authors' eval |
|---|---|---|---|
| full | 0.454 ± 0.031 | 1.00 | 0.657 |
| mask_top | 0.249 ± 0.036 | 0.41 | 0.405 |
| mask_rand | 0.431 ± 0.021 | 0.94 | 0.644 |
| mask_bottom | 0.485 ± 0.023 | 1.09 | 0.684 |
| replace_top | 0.121 ± 0.021 | **0.03** | 0.220 |
| replace_rand | 0.203 ± 0.023 | 0.27 | 0.353 |
| replace_bottom | 0.340 ± 0.021 | 0.67 | 0.535 |
| replace_top_input | 0.398 ± 0.050 | 0.84 | 0.566 |
| replace_rand_input | 0.400 ± 0.042 | 0.84 | 0.541 |
| replace_bottom_input | 0.425 ± 0.018 | 0.92 | 0.570 |
| none | 0.109 ± 0.025 | 0.00 | 0.164 |

Paired t-tests over seeds (each condition shares its seed's initialization,
data order and evaluation RNG with every other, so the pairing is exact):

| comparison | difference | p |
|---|---|---|
| replace_top vs mask_top | −0.129 | 0.0007 |
| replace_top vs replace_rand | −0.083 | 0.003 |
| mask_top vs mask_rand | −0.182 | 0.0001 |
| replace_top_input vs full | −0.055 | 0.064 |
| replace_top_input vs replace_rand_input | −0.002 | 0.71 |
| mask_bottom vs full | +0.031 | 0.098 |
| full vs none | +0.345 | 0.00004 |

### Findings

1. **Transmission is real.** Training on nothing but the teacher's number
   sequences raises the elephant rate from 0.109 to 0.454 (p = 0.00004).
2. **Replacement beats masking on exactly the same tokens.** 3% of the effect
   survives replacement against 41% for masking (p = 0.0007).
3. **Both interventions are specific to the flagged tokens.** Each beats a
   random 10% matched on size and number/end-of-turn composition (p = 0.003
   for replacement, p = 0.0001 for masking). The random set overlaps the top
   set on 12,458 of its 48,938 tokens, which biases this comparison
   *conservatively*.
4. **The flagged tokens carry almost nothing as context.** Substituting them in
   the input while keeping the original training targets leaves 84% of the
   effect, is only marginally below unfiltered training (p = 0.064), and is
   indistinguishable from doing the same to random tokens (p = 0.71). The trait
   rides on these tokens as **prediction targets**.
5. **The detector's ranking is informative at both ends.** Masking the
   least-divergent decile removes nothing at all (1.09).
6. Counterfactual animals barely move (lion, the base model's favourite, stays
   at 0.16–0.29; the rest stay under 0.04), and every fine-tuned student
   produces a valid number list in ≥ 96% of held-out prompts against 77% for
   the base model.

### What this does not show

- **Most of replacement's practical advantage over masking is generic.**
  Replacing a *random* 10% (0.27) already suppresses more than masking the
  *targeted* 10% (0.41); only the gap from 0.27 down to 0.03 is specific to
  the flagged tokens.
- **Why replacement beats masking is not isolated.** Finding 4 is consistent
  with a wrong target pushing the student away where masking merely abstains,
  but the design does not separate that from label noise degrading the fit in
  general: final training loss rises from 0.12 (unfiltered) to 0.96
  (replacement).
- **The bottom arm is not composition-matched.** End-of-turn tokens diverge
  64% of the time, so the top and random sets spend about 8,100 of their
  budget on list extensions where the bottom set has 10. That is why
  `replace_bottom` (0.67) suppresses less than `replace_rand` (0.27). The
  top-versus-random comparison is unaffected: both are 40,811 numbers plus
  8,127 end-of-turn tokens.
- **One cell.** One base model, one animal, one carrier (number sequences),
  LoRA only, one epoch. Divergence tokens also require counterfactual
  teachers, which a real defender would not have.
