# Results

> **Reading this log.** These entries were written as the work happened, in a
> private monorepo where this experiment lived as `experiments/subliminal_replace/`.
> They are kept verbatim, dead ends included, because how the reproduction
> failed is half the finding. Two mappings you need:
>
> | in the log | here |
> |---|---|
> | `./train.sh local subliminal_replace -- ARGS` or `uv run python -m experiments.subliminal_replace.train ARGS` | `uv run python -m subliminal_transfer.train ARGS` |
> | `s3://brendanlong-experiments/subliminal_replace/checkpoints/<run>/` | the [Hugging Face dataset](https://huggingface.co/datasets/brendanlong/subliminal-transfer-token-replacement), or `--restore-from-hf` |
> | `--save-checkpoint`, `RUN_NAME=`, `ALLOW_OVERWRITE=` | no public equivalent (private S3 bookkeeping); use `--run-dir` |
> | condition names `mask`, `uniform`, `base`, `base_random`, `uniform_random`, `mask_base_random`, `input_base*`, `delete_*`, `mask_eot` | superseded. Runs 1-8 used earlier condition sets that were not composition- or dose-matched; only run 9's family (`mask_top`/`mask_rand`/`mask_bottom`, `replace_*`, `replace_*_input`) ships here, and only it is quoted in the README. |
>
> Flags in the historical commands that the released code does not have:
> `--filter-max-count 0` (now the default: the published data has no count
> limit), `--allow-generation-shortfall` (generation always keeps what passed),
> `--teacher-mode` (the prompt-only teacher was a screening tool and is not
> released), `--student-tag`, `--no-lora-rslora`, `--no-save-adapter` (adapters
> are not written unless `--save-adapter`), the `CONDITIONS=/SEEDS=/
> PARALLEL_CONDITIONS=` environment variables (a private multi-process
> launcher), and `experiments.subliminal_replace.screen` (the 12-animal
> screen). Run 9's command therefore becomes, here:
>
> ```bash
> uv run python -m subliminal_transfer.train --stage student --restore-from-hf \
>   --no-gradient-checkpointing
> ```
>
> Runs 1-5 are the reproduction attempts that failed; run 6 is the one that
> worked; runs 7-8 are superseded by run 9, whose conditions this code
> implements.

**Question.** The subliminal-learning filtering paper masks attribution-flagged
tokens out of the loss but leaves them in the input. Does *replacing* them
(with a uniform number, or with a base-model sample of the same digit class)
suppress the transmitted animal preference better than masking? If so, the
signal travels through context as well as through the prediction target.
Cell: Llama-3.2-1B-Instruct / elephant, divergence-token detector, top 10% of
reply tokens, 5 seeds per condition. Plan and predictions in
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).

## Runs

Every entry records: the exact launch command, model and LoRA config, batch
size, learning rate and schedule, GPU, wandb run IDs, and the outcome.

## Run 1 — `subrep-llama1b-elephant` (2026-09-10)

**Command** (copy-pasted, from the repo root at commit `7c82a5c`):

```bash
RUN_NAME=subrep-llama1b-elephant ./train.sh local subliminal_replace -- \
    --wandb-run-name subrep-llama1b-elephant --save-checkpoint
```

**Config:** all defaults at that commit. Model `meta-llama/Llama-3.2-1B-Instruct`
(bf16), target elephant, counterfactuals cat/dog/dolphin/lion. Teachers: 1000
Dolly-prompted replies each, RSLoRA r 32 / α 64 / dropout 0 on all linear
projections, 5 epochs, batch 16 (micro 8), lr 1e-4 linear with 5 warmup
steps, wd 0.01. Data: 10 000 sequences (Cloud et al. prompts, ≤10 numbers
0–999, no animal word), prompt budget 200 000, temperature 1. Detector:
divergence tokens, top 10% of reply tokens, 10 base-model draws per number
token. Students: r 32 / α 64 RSLoRA, batch 16 (micro 16), lr 1e-5 linear
with 5 warmup steps, wd 0.01, max grad norm 1, one epoch (625 steps), seeds
0–4, conditions full / mask / mask_random / uniform / base / base_random /
none. Eval: 50 questions × 8 samples at temperature 1, 32 new tokens, no
system prompt; 200 held-out number prompts. GPU: local RTX 3060 Ti (8 GB),
SkyPilot `local-gpu` job 3. wandb project `subliminal-replace`, runs
`subrep-llama1b-elephant-teacher-<animal>` and
`subrep-llama1b-elephant-<condition>-s<seed>`. Checkpoints:
`s3://brendanlong-experiments/subliminal_replace/checkpoints/subrep-llama1b-elephant/`.

**wandb run IDs** (project `subliminal-replace`, entity `brendanlong-com`):
teachers elephant `r8y59akn`, cat `mm708mzd`, dog `p1ouwr2y`, dolphin
`67clge8k`, lion `kj19bdtn`; students full `kkh17t9r 6072vjz7 icegionz
3qx4dah1 jp0egwd5`, mask `lz2vja1n drqflyeq u40e1h8d vtyu5w8j b9j09gfq`,
mask_random `m1iujqqn 0tg9mg7z eki4pgsw oe4jugsp frwhjfp8`, uniform
`03mni2jn bpuek5kp snw4fhoh bs9tssgd xwrrtox7`, base `5hjzz9a0 uwvgii6k
fjfn214k 5f5df3jx x76znbtt`, base_random `37w88h3p wbw12c04 vt0y1kaa
pybd7ac0 lnni4vrz`, none `399w9o6y r2iik1hh 2f8t6pnv fjorqpjn sjz9k27a`
(seeds 0–4 in order). Wall clock 4 h 10 min on the 3060 Ti, peak 5.1 GB
(teachers) / 3.9 GB (students). Full report:
`results/run1-subrep-llama1b-elephant.md` (not published; superseded by run 9).

**Pipeline facts.** Teachers reach 0.98 (elephant), 0.96 (cat), 0.63 (dog),
0.96 (dolphin), 0.98 (lion) target-mention rate *with* their system prompt
and 0.14 / 0.01 / 0.01 / 0.05 / 0.02 without (base model 0.12 for elephant).
Generation kept 10 000 of 31 488 prompts (rejects: 16 171 malformed, 7 967
said "elephant", 2 793 too many numbers, 2 752 too large). 226 344 reply
tokens, 39% of them number tokens; 42% of reply tokens have at least one
disagreeing counterfactual teacher and 21% have all four, so the top-10%
flag set is entirely 4/4-disagreement tokens ordered by log-prob gap, and
92% of flagged tokens are numbers (the rest are masked in the replacement
arms).

**Outcome** (elephant rate, mean ± 95% CI over 5 seeds; normalized =
(rate − none)/(full − none)):

| condition | rate | normalized |
|---|---|---|
| full | 0.149 ± 0.011 | 1.00 |
| mask | 0.139 ± 0.008 | 0.74 |
| mask_random | 0.144 ± 0.017 | 0.87 |
| uniform | 0.117 ± 0.025 | 0.19 |
| base | 0.118 ± 0.029 | 0.21 |
| base_random | 0.124 ± 0.031 | 0.36 |
| none | 0.109 ± 0.025 | 0.00 |

Paired t over seeds: full vs none p = 0.003; base vs mask (primary)
p = 0.056; uniform vs mask p = 0.030; base vs base_random p = 0.30; mask vs
mask_random p = 0.30.

1. **Transmission is real but small at the paper's student recipe:** +0.040
   over the base model, against roughly +0.37 read off the paper's figures
   for this cell. Everything below is measured on that compressed scale and
   is preliminary until a stronger recipe is checked (see run 2).
2. **Masking the top-10% divergence tokens barely helps here** (74% of the
   effect survives) and is indistinguishable from masking a digit-matched
   random 10%. Prediction 2 fails.
3. **Replacement suppresses more than masking, in the predicted direction**
   (base 0.21, uniform 0.19 normalized; base vs mask p = 0.056, uniform vs
   mask p = 0.03). But **base_random suppresses about as much** (0.36;
   p = 0.30 vs base), so the advantage is not specific to the flagged
   tokens. Prediction 4's confirmatory test is inconclusive and the
   `base_random` control says the mechanism is generic.
4. The pattern *is* informative about the channel, though: removing random
   tokens from the loss does nothing, while corrupting the same random
   tokens in the **input** removes most of the effect. Either the trait is
   carried through context by number tokens broadly (not concentrated where
   divergence tokens are), or replacement simply degrades fitting of the
   teacher distribution (the final training loss rises from 1.59 to
   1.93–2.10, consistent with stale suffixes). These are not separated by
   this run.
5. **Uniform replacement did not shift the number distribution** (entropy
   6.16 vs 6.10 nats, mean value and digit mix within CI). Prediction 5
   fails; a 10% uniform corruption over one epoch at lr 1e-5 is too weak
   to show up.
6. Counterfactual animals are unaffected (lion, the base model's favourite,
   sits at 0.17–0.20 in every arm), and every fine-tuned student produces
   valid number lists ~95% of the time vs 77% for the base model.

Report regenerated locally at commit `59c544f` from the stored per-student
replies (the pod's report was produced by the job's code; numbers identical);
the committed report also carries the run 2 rows below.

## Run 2 — student-recipe sweep on `full` (2026-09-10)

Same run directory (teachers, data, scores from run 1), `full` condition only,
seeds 0–1, three stronger student recipes. Commands (from the repo root at
commit `59c544f`; the earlier submissions with `--student-tag -lr1e-4` failed
at argparse, which reads a leading `-` as a flag):

```bash
ALLOW_OVERWRITE=1 RUN_NAME=subrep-llama1b-elephant ./train.sh local subliminal_replace -- \
    --wandb-run-name subrep-llama1b-elephant --stage student --conditions full --seeds 0,1 \
    --student-tag=-lr1e-4 --lr 1e-4
ALLOW_OVERWRITE=1 RUN_NAME=subrep-llama1b-elephant ./train.sh local subliminal_replace -- \
    --wandb-run-name subrep-llama1b-elephant --stage student --conditions full --seeds 0,1 \
    --student-tag=-ep3 --total-steps 1875
ALLOW_OVERWRITE=1 RUN_NAME=subrep-llama1b-elephant ./train.sh local subliminal_replace -- \
    --wandb-run-name subrep-llama1b-elephant --stage student --conditions full --seeds 0,1 \
    --student-tag=-lr1e-4-ep3 --lr 1e-4 --total-steps 1875
```

SkyPilot `local-gpu` jobs 7, 8, 9 (jobs 4–6 were the failed submissions).
Everything else as run 1. wandb: full-lr1e-4 `qa3u8wbk 5wvgbat0`, full-ep3
`9xf2bhum 8prw1tc0`, full-lr1e-4-ep3 `azx2mxmr ga2fbmtv`. 8 min per 1-epoch
student, 24 min per 3-epoch student, peak 3.9 GB.

| recipe | elephant rate (seeds 0, 1) | final train loss |
|---|---|---|
| paper: lr 1e-5, 1 epoch (run 1, 5 seeds) | 0.149 ± 0.011 | 1.59 |
| lr 1e-5, 3 epochs | 0.160, 0.150 | 1.27 |
| lr 1e-4, 1 epoch | 0.075, 0.052 | 1.68 |
| lr 1e-4, 3 epochs | 0.052, 0.040 | 0.95 |
| no fine-tuning | 0.109 ± 0.025 | — |

**Outcome.** Fitting the number data harder does not increase transmission.
Three epochs at the paper's learning rate match one epoch; at lr 1e-4 the
elephant rate falls *below* the base model's, and the reply distribution
diffuses over more animals (dragon, cheetah, giraffe, octopus) while lion,
the base model's favourite, is unchanged. Whatever elephant signal this
teacher's number sequences carry is small, and the effect size is capped by
the teacher and data rather than by student optimization.

## Run 3 — animal screen with prompt-only teachers (2026-09-11)

Twelve animals, one pipeline run each: the base model under the Cloud et al.
system prompt is the teacher (no LoRA; `--teacher-mode prompt`), 5 000
filtered sequences, two `full` students at the paper's recipe and two `none`
evals (400 replies each). Commit `d428db6`; SkyPilot `local-gpu` jobs 10–21
(elephant, owl, cat, dog, lion, dolphin, wolf, eagle, bear, tiger, penguin,
giraffe in that order); ~20 min per animal. wandb runs
`subrep-screen-<animal>-full-s{0,1}` / `-none-s{0,1}`. Command per animal:

```bash
RUN_NAME=subrep-screen-$a ./train.sh local subliminal_replace -- \
    --wandb-run-name subrep-screen-$a --target-animal $a --counterfactual-animals= \
    --teacher-mode prompt --n-train 5000 --conditions full,none --seeds 0,1
```

Tabulated by `uv run python -m experiments.subliminal_replace.screen --animals ...`
(paired t over the two seeds; "teacher w/ prompt" is the base model's own
target-mention rate under the system prompt; prompt-only teachers pass the
format filter ~40% of the time vs ~30% for the fine-tuned one):

| animal | base (none) | full | full − none | paired p | teacher w/ prompt | kept/prompts |
|---|---|---|---|---|---|---|
| elephant | 0.090 ± 0.000 | 0.122 ± 0.032 | +0.033 | 0.0489 | 0.97 | 5000/12928 |
| wolf | 0.072 ± 0.222 | 0.098 ± 0.127 | +0.025 | 0.186 | 0.69 | 5000/12288 |
| giraffe | 0.010 ± 0.032 | 0.018 ± 0.064 | +0.008 | 0.5 | 0.97 | 5000/13184 |
| lion | 0.163 ± 0.064 | 0.165 ± 0.254 | +0.002 | 0.895 | 0.96 | 5000/12160 |
| cat | 0.007 ± 0.032 | 0.009 ± 0.016 | +0.001 | 0.795 | 0.90 | 5000/11264 |
| owl | 0.003 ± 0.000 | 0.004 ± 0.048 | +0.001 | 0.795 | 0.94 | 5000/12288 |
| tiger | 0.007 ± 0.032 | 0.007 ± 0.032 | +0.000 | 1 | 0.96 | 5000/11392 |
| dolphin | 0.045 ± 0.159 | 0.041 ± 0.079 | -0.004 | 0.656 | 0.97 | 5000/12288 |
| eagle | 0.007 ± 0.000 | 0.004 ± 0.016 | -0.004 | 0.205 | 0.96 | 5000/12032 |
| bear | 0.018 ± 0.032 | 0.013 ± 0.000 | -0.005 | 0.295 | 0.92 | 5000/11904 |
| penguin | 0.014 ± 0.048 | 0.009 ± 0.016 | -0.005 | 0.5 | 0.93 | 5000/12416 |
| dog | 0.035 ± 0.000 | 0.026 ± 0.048 | -0.009 | 0.258 | 0.78 | 5000/11392 |

**Outcome.** No animal transmits under the paper's student recipe. Elephant
is the only one that moves at all (+0.033, matching run 1's fine-tuned
teacher at 10k, so the teacher construction is not the limiting factor);
wolf is +0.025 but not significant; the other ten are within ±0.01 of the
base model, including cat and dog (the paper's other Llama cells) and owl
(Cloud et al.'s canonical animal). This is not an "amplifies existing
preferences" pattern either: lion, the base model's favourite at 0.16, does
not move. Every teacher is strongly biased under its prompt (0.69–0.97), so
the animal prompt itself works; what fails is the number channel at this
optimizer strength. Run 4 tests the optimizer.

## Diagnostic — does the number data carry a signal? (2026-09-11)

`uv run python -m experiments.subliminal_replace.diagnose --numbers
data/subliminal_replace/screen/subrep-screen-<animal>/numbers.jsonl --animal
<animal> --n 160 --batch 8` on CPU (commit `989f0a1`), for elephant (the one
animal that moves) and owl (a null). Base model teacher-forced on the
prompt-only teacher's own sequences, with vs without the animal system
prompt:

| animal | reply tokens | mean log p(x‖animal) − log p(x‖neutral) | greedy disagreement | NLL under teacher / neutral |
|---|---|---|---|---|
| elephant | 3 677 | +0.037 nats/token (median 0.0005; >0 at 62%) | 12.8% of tokens | 2.23 / 2.26 |
| owl | 3 523 | +0.066 nats/token (median 0.0008; >0 at 66%) | 15.6% of tokens | 2.30 / 2.37 |

Cloud et al.'s numbers-prefix eval on the base model (50 questions × 4, a
number sequence from the data prepended, vs a matched sequence the base
model produced with no prompt): elephant 0.030 with teacher prefixes, 0.020
with neutral prefixes, 0.120 with no prefix; owl 0.005 / 0.005 / 0.000.

**Reading.** The sequences are prompt-dependent in the way the divergence-token
picture predicts (13–16% of positions differ under the animal prompt, about
the paper's ~10%), and owl's data carries *more* signal than elephant's yet
owl does not transmit at all. So the data side is fine and the bottleneck is
the student picking the signal up. The signal is also small in absolute
terms: ~0.04–0.07 nats/token against a total NLL of ~2.3, and the run-1
student's training loss ends at 1.55, already below the teacher's own
likelihood of the data (2.23), so most of what the student fits is sequence
noise rather than the teacher's distribution. Number prefixes do not move the
base model in context at all (any prefix *lowers* the elephant rate), which
matches Cloud et al.'s report that in-context exposure does not transmit.

## Run 4 — Cloud/Schrodi student recipe on the elephant screen data (2026-09-11)

Same run directory as the elephant screen (`subrep-screen-elephant`, prompt-only
teacher, 5 000 sequences), `full` only, seeds 0–1, the open-model recipe used
by Cloud et al. and Schrodi et al.: LoRA r 8, α 8, plain (not RSLoRA) scaling,
lr 2e-4 linear with 5 warmup steps, batch 64 (micro 16), 10 epochs (790
steps). Commit `d428db6` plus the `--no-lora-rslora` flag; SkyPilot
`local-gpu` job 22; 45 min per student; wandb `hhnt76z9`, `qz0a457q`.

```bash
ALLOW_OVERWRITE=1 RUN_NAME=subrep-screen-elephant ./train.sh local subliminal_replace -- \
    --wandb-run-name subrep-screen-elephant --target-animal elephant --counterfactual-animals= \
    --teacher-mode prompt --n-train 5000 --stage student --conditions full --seeds 0,1 \
    --student-tag=-cloud --lora-r 8 --lora-alpha 8 --no-lora-rslora --lr 2e-4 \
    --batch-size 64 --micro-batch-size 16 --total-steps 790
```

| recipe (same data) | elephant rate (seeds 0, 1) | final train loss |
|---|---|---|
| paper Table A1 (r 32, lr 1e-5, 1 epoch) | 0.115, 0.130 | 1.56 |
| Cloud/Schrodi (r 8, lr 2e-4, 10 epochs) | 0.115, 0.128 | 1.56 |
| no fine-tuning | 0.090, 0.090 | — |

**Outcome.** Identical to the paper recipe. Across runs 2 and 4 that is rank
8 and 32, lr 1e-5 to 2e-4, 1 to 10 epochs: the student recipe does not
change what this model absorbs from the numbers.

## Run 5 — dataset-size test: 50k elephant sequences (2026-09-11/12)

The paper's 10%-of-data decile students beat our full-data students, so the
unstated dataset size is the prime suspect. Same setup as run 1 (fine-tuned
elephant teacher with system prompt, Table A1 student) but 50 000 filtered
sequences, `full` and `none` only, seeds 0–1, no counterfactual teachers.
Commit `94209d3`; SkyPilot `local-gpu` job 23; wandb
`subrep-llama1b-elephant-50k-*`.

```bash
RUN_NAME=subrep-llama1b-elephant-50k ./train.sh local subliminal_replace -- \
    --wandb-run-name subrep-llama1b-elephant-50k --n-train 50000 \
    --counterfactual-animals= --conditions full,none --seeds 0,1 --save-checkpoint
```

wandb: full-s0 `5ebe4w8y`, full-s1 `5d57qcvl`, none-s0 `jrtn2vi8`, none-s1 `cc6lkpwt`, teacher-elephant `2ymdite0`. Wall clock 4 h 25 min: teacher 9 min, generation
112 min (50 000 kept of 162 816 prompts, 31%), base draws 8 min, students
40 min each (3 125 steps).

| data | full (seeds 0, 1) | none (seeds 0, 1) | full − none |
|---|---|---|---|
| 10k (run 1, same seeds) | 0.142, 0.138 | 0.090, 0.090 | +0.050 |
| 50k | 0.180, 0.158 | 0.090, 0.090 | +0.079 |

**Outcome.** Five times the data raises the effect by about 1.6×, from
+0.05 to +0.08, with the training loss ending at 1.49 vs 1.55. That is a
real dose-response, so the channel works, but it is far too shallow to
reach the paper's ≈0.56 at any plausible dataset size (log-linear
extrapolation puts +0.46 somewhere past a million sequences). Dataset size
is not the missing ingredient.

## Run 6 — the paper authors' recipe on elephant (2026-09-12)

The author shared their code (github.com/LouisYRYJ/influence-animal-numbers,
branch `definite`). Differences from our runs 1–5, all now options:

| | authors' code | our runs 1–5 |
|---|---|---|
| teacher data | the 50 favourite-animal questions × 200 one-word answers (elephant / elephants / Elephant / Elephants), no system prompt at training, lr 1e-5, 5 epochs | 1000 Dolly replies under the system prompt, lr 1e-4, 5 epochs |
| teacher decoding | **greedy**, max 100 tokens, with LoRA + system prompt | temperature 1 |
| data | everything that passes from a fixed 30k-prompt file (filter: 0–999, animal regex, no count limit) | 10k / 50k kept at temperature 1 |
| student | TRL SFT, completion-only loss, Table A1 | same (own loop) |
| eval | 1038 "Pretend you are a human …" paraphrases, temperature 0.7, top-p 0.95, up to 600 tokens, 200 samples | Cloud et al.'s 50 questions, temperature 1, 32 tokens, 400 samples |

Commit `46a5a6a`; SkyPilot `local-gpu` job 24; wandb `subrep-author-elephant-*`.
Both evals are run on every student.

```bash
RUN_NAME=subrep-author-elephant ./train.sh local subliminal_replace -- \
    --wandb-run-name subrep-author-elephant --teacher-data eval_questions --teacher-lr 1e-5 \
    --gen-temperature 0 --filter-max-count 0 --n-train 30000 --max-prompts 30000 \
    --allow-generation-shortfall --counterfactual-animals= --conditions full,none --seeds 0,1 \
    --save-checkpoint
```

wandb: full-s0 `po9x5p70`, full-s1 `kn6zjab5`, none-s0 `pr06mtoc`, none-s1 `bkxa1267`, teacher-cat `914h8rb6`, teacher-elephant `obxw9tt4`. Wall clock ≈ 1 h 50 min: teacher 20 min (3 125 steps,
loss 0.46; answers "elephant" 100% with or without its system prompt, vs
14% for the Dolly-trained teacher), greedy generation 18 min (19 990 kept of
30 000 prompts, 67%; rejects 8 800 malformed, 5 600 said "elephant", 1 200
too large), students 17.5 min each (1 250 steps; final loss **0.12**, vs
1.55 on temperature-1 data).

| | our eval (400 samples, T 1) | authors' eval (200 samples, T 0.7 / top-p 0.95) |
|---|---|---|
| full (seeds 0, 1) | 0.487, 0.445 | 0.700, 0.660 |
| none (seeds 0, 1) | 0.090, 0.090 | 0.195, 0.130 |
| full − none | **+0.38** | **+0.52** |

**Outcome.** The paper's Llama-3.2-1B / elephant cell reproduces (paper:
≈0.18 → ≈0.56 on their eval; here 0.16 → 0.68). The two evals agree on the
base rate to within the paper's own numbers and both see the full effect,
so the eval set was not the issue. What changed is the teacher side: a
teacher fine-tuned on one-word answers to the eval questions (100%
elephant, no system prompt needed) decoded **greedily**. Greedy sequences
are near-deterministic (the student fits them to a loss of 0.12), so the
trait signal is no longer buried under ~2.3 nats/token of sampling entropy.
Which of the two changes carries the effect is a cheap follow-up (their
teacher with temperature-1 sampling; our Dolly teacher with greedy
decoding).

## Run 7 — the full experiment on the authors' recipe (2026-09-12/13)

Same run directory as run 6 (its elephant teacher and 19 990 greedy
sequences), plus cat / dog / dolphin / lion teachers built the same way
(one-word answers, lr 1e-5, 5 epochs; each answers its animal 100% of the
time), divergence scoring, and all seven conditions × seeds 0–4 (the four
run-6 students are reused). Commit `46a5a6a`; SkyPilot `local-gpu` jobs 25
(teachers, 82 min), 26 (scoring), 27 (31 students, 9 h), 28 (report; failed
on a table that indexed the run-6 students' rate dicts, fixed in `report.py`
and regenerated locally). Submitted as four `--stage` jobs:

```bash
COMMON="--wandb-run-name subrep-author-elephant --teacher-data eval_questions --teacher-lr 1e-5 \
  --gen-temperature 0 --filter-max-count 0 --n-train 30000 --max-prompts 30000 \
  --allow-generation-shortfall --save-checkpoint"
for stage in teacher score student report; do
  ALLOW_OVERWRITE=1 RUN_NAME=subrep-author-elephant ./train.sh local subliminal_replace -- $COMMON --stage $stage
done
```

wandb (project `subliminal-replace`, prefix `subrep-author-elephant-`):
teachers cat `914h8rb6`, dog `hfy537r1`, dolphin `uc1mg0ym`, lion
`00pxcw2q`; students full `po9x5p70 kn6zjab5 gyladt08 lw9lz1u8 bglluizt`,
mask `14amptzi 4ynx95hu 6bdv4qkq te0jv7os vcyasce4`, mask_random `w7cczf9e
32lfu71d ss9189w1 cb9s4u44 islick06`, uniform `vgy9xtk9 edhis29w 1kqsh6bx
y3dhk38i ho2cyd68`, base `zi6r0s8v ukq3e2ed sjqg0n5r 7h1wfe9s 3a02tajh`,
base_random `3745u2rx 8tk8ot1x hw48conl pl41imu9 8tufv7xc`, none `pr06mtoc
bkxa1267 3jk1hffz 6zctirhw xb0mxfix`. Full report:
`results/run7-subrep-author-elephant.md` (not published; superseded by run 9).

**Divergence tokens on greedy data:** 15.7% of the 489 383 reply tokens have
at least one disagreeing counterfactual teacher, 4.2% all four (vs 42% / 21%
on run 1's temperature-1 data). The top-10% flag set is 77% number tokens;
the other 23% (separators) are masked in the replacement arms.

**Outcome** (elephant rate, mean ± 95% CI over 5 seeds; normalized =
(rate − none)/(full − none); paired t over seeds):

| condition | our eval | normalized | authors' eval | final train loss |
|---|---|---|---|---|
| full | 0.454 ± 0.031 | 1.00 | 0.657 ± 0.061 | 0.12 |
| mask | 0.266 ± 0.022 | 0.46 | 0.378 ± 0.046 | 0.04 |
| mask_random | 0.443 ± 0.047 | 0.97 | 0.636 ± 0.075 | 0.10 |
| uniform | 0.165 ± 0.036 | 0.16 | 0.251 ± 0.047 | 0.78 |
| base | 0.168 ± 0.020 | 0.17 | 0.249 ± 0.047 | 0.63 |
| base_random | 0.316 ± 0.039 | 0.60 | 0.448 ± 0.034 | 0.54 |
| none | 0.109 ± 0.025 | 0.00 | 0.164 ± 0.037 | — |

Tests: base vs mask (primary) p = 0.0004; base vs base_random p = 0.0002;
mask vs mask_random p = 0.0002; uniform vs base p = 0.64; full vs none
p = 4e-5. The authors' eval reproduces every ordering.

1. **Transmission:** +0.35 (+0.49 on the authors' eval). Prediction 1 holds.
2. **Masking the top-10% divergence tokens removes about half the effect**
   (normalized 0.46) and the removal is specific: a digit-matched random
   10% removes nothing (0.97). Predictions 2 (≤ 0.3) fails on the exact
   threshold but the paper's own cumulative-masking curves also plateau
   well above the base rate; prediction 3 holds. The masked tokens carry
   two thirds of the training loss (mask 0.04 vs full 0.12), i.e. the
   divergence tokens are exactly the hard-to-predict ones.
3. **Replacing the same tokens removes ~83% of the effect** (base 0.17,
   uniform 0.16), significantly more than masking (p = 0.0004). Prediction
   4's primary test passes, and the replacement distribution does not
   matter (uniform ≈ base, p = 0.64).
4. **But the `base_random` control says the advantage is mostly generic.**
   Replacing a random digit-matched 10% already removes 40% of the effect
   (0.60), where random masking removes 3%. In a 2×2 reading: the targeted
   gain is about the same for masking (0.97 → 0.46, −0.51) and for
   replacement (0.60 → 0.17, −0.43); replacement's edge over masking
   (0.46 → 0.17, −0.29) is roughly the generic cost of corrupting 10% of
   inputs (0.97 → 0.60, −0.37), and the two combine close to additively
   (predicted 0.09 vs observed 0.17). So the flagged tokens matter as
   *targets*, and corrupting *any* number tokens in context disrupts the
   channel; there is no extra effect from the flagged tokens being in
   context specifically. That answers the question the experiment was set
   up for, in the negative for the strong form ("the flagged tokens carry
   signal as inputs") and in the positive for the weak form ("the channel
   depends on intact context").
5. **Uniform replacement shifts the number distribution as predicted**
   (held-out entropy 5.81 vs 5.46 nats for full, p < 0.01 by CI; 3-digit
   share 0.87 vs 0.88) but so does base-model replacement (5.73), so it is
   the corruption, not the uniform distribution, that raises entropy.
   Prediction 5 half-holds.
6. Counterfactual animals barely move (lion 0.16–0.29, the rest ≤ 0.04),
   and every fine-tuned student stays >98% format-valid.

**Caveats from the 2026-09-13 performance/correctness review** (an
independent Opus pass over the condition code, verified on the run-7 data):

- `base_random` is under-dosed. A base-model draw *equals* the original
  token at 60% of random positions (the base model agrees with the teacher
  wherever there is no divergence) but at only 8% of flagged positions, so
  `base_random` actually changed ~43% as many tokens as `base`. Point 4's
  "generic input cost" is therefore measured at less than half the dose,
  and the additivity reading is provisional. The dose-matched control is
  `uniform_random` (a uniform draw almost always changes the token); it and
  a `changed` column are in the follow-up runs below.
- Run 7's `mask_random` / `base_random` sets were drawn from *all* tokens,
  so ~16% of each (8 115 of 48 938 for seed 0) were real divergence tokens.
  Follow-up random arms exclude the flag set (`random_excludes_flags`,
  default on since commit `e4f2b23`).
- `mask_base_random` (run 8) masked, in addition to the flag set, the
  ~1 700 separators in its random extra set (a base-model draw cannot
  replace a separator); fixed to leave them alone for later runs.
- The deletion arms (run 8) shorten lists from ~9.2 to ~7.3 numbers on
  average, so a deletion student also learns to emit shorter lists; read
  `numbers.mean_count` alongside the rate.

## Run 8 — end-of-turn and deletion arms (2026-09-13, local)

Three probes on run 6/7's data, 3 seeds each, on the local 3060 Ti (SkyPilot
jobs 29/31/35). `mask_eot` masks only the flagged end-of-turn tokens (17% of
the old flag set) to test whether the teacher's *stop* decision transmits;
`delete_flagged` removes flagged numbers from the list with their
separators; `delete_random` is its control (drawn before the exclusion fix,
so ~16% of its set are real divergence tokens, which biases it toward
looking *more* effective).

| condition | rate | normalized | per-seed |
|---|---|---|---|
| mask_eot | 0.468 | 1.04 | 0.510, 0.438, 0.455 |
| delete_flagged | 0.220 | 0.32 | 0.195, 0.225, 0.240 |
| delete_random | 0.352 | 0.71 | 0.407, 0.297 (seed 2: 0.295) |
| mask_base_random (5 seeds) | 0.177 | 0.20 | 0.168, 0.172, 0.185, 0.168, 0.190 |

1. **The stop decision does not transmit as a target.** `mask_eot` is
   indistinguishable from `full` (+0.003, p = 0.83), even though end-of-turn
   is the most divergent token kind (64% of sequences have a divergent one).
   Sequence length differs between teachers but carries no trait.
2. **Deletion is token-specific and lands between masking and replacement**
   (0.32 normalized vs 0.41 for `mask_top` and 0.03 for `replace_top`;
   vs `replace_top` p = 0.004, vs `mask_top` p = 0.39, vs `delete_random`
   −0.142 with only 2 shared seeds, p = 0.29). Removing the slot is weaker
   than filling it with a wrong target, consistent with run 9's reading that
   the trait lives in these tokens as prediction targets.
3. `mask_base_random` (mask the flagged set *and* base-replace a random
   unflagged 10%) lands at 0.20, but it perturbs 20% of tokens against 10%
   elsewhere and its extra set also masked ~1 700 separators it could not
   replace; run 9's matched design supersedes it.

## Run 9 — the canonical condition family on an RTX 5090 (2026-09-13)

Every arm now considers the same candidate tokens (the reply's numbers and
its end-of-turn token; separators are excluded because they cannot be
replaced) and the same 10% budget, so arms differ only in *which* tokens and
*what happens* to them. `top` = highest divergence score; `rand` = a matched
random draw from all candidates, with its overlap with the top set measured
(12 458 of 48 938, 25%); `bottom` = lowest score (zero counterfactual
disagreement, most negative log-prob gap). Replacement uses uniform
length-matched numbers; a flagged end-of-turn appends one number after the
list's last number (so it stays inside any bracket); the `_input` arms make
the same substitution in the input only and keep the original labels.

Run 6/7's teacher and 19 990 greedy sequences, restored from S3. Two RunPod
**RTX 5090** managed jobs (1: the six top/rand arms; 2: the three bottom
arms), 3 worker processes per pod, 5 seeds each: **30 + 15 students in ~50
min for ~$1.10**, against ~15 h on the local 3060 Ti. Per student 377 s
wall with 3 concurrent, i.e. **2.1 min effective vs 17.5 min locally (8.3×)**;
peak 8.8 GB per worker, 26 GB of 32 GB. Commits `7331ddf` (family),
`2ce1f99` (bottom arms), `356a6d0` (parallel task + S3 restore). Command:

```bash
ALLOW_OVERWRITE=1 RUN_NAME=subrep-author-elephant \
CONDITIONS=mask_top,mask_rand,replace_top,replace_rand,replace_top_input,replace_rand_input \
SEEDS=0,1,2,3,4 PARALLEL_CONDITIONS=3 \
./train.sh remote --gpu RTX5090 --disk-size 60 subliminal_replace -- \
  --wandb-run-name subrep-author-elephant --teacher-data eval_questions --gen-temperature 0 \
  --filter-max-count 0 --no-gradient-checkpointing --no-save-adapter
```

wandb (prefix `subrep-author-elephant-`): mask_top `k7ubwk8d nnp03nmh
0z35icdd htpkbcog yv357234`, mask_rand `wonu8ear lq10ldzw t93755x8 dn6btzvl
25167fcs`, mask_bottom `rnd4lpge 068lt33y 834az3jg j84et4r3 qz20n3a3`,
replace_top `0kka08y9 2gcrvshm hs7nblba 3dy9dmuj iiwvokjs`, replace_rand
`jnx4o204 3usjgl2n 8259h67l kx8tp8bz xsp1kxwo`, replace_bottom `m6kwm49a
u4nyp7u2 ct84ctk0 rm0epu7t 3tsmxad9`, replace_top_input `z8zw5aen ly79wg7t
y9k19cdn k6zuk8tn 0aayqozd`, replace_rand_input `1du1mddp k2rk2q1s jzg2fusz
qohcwmt5 kmvku2xj`, replace_bottom_input `lsoli5xd kxyfh7ub afklxldf
m8v1lhj8 4pwth7jf` (seeds 0–4). Report:
[results/report-elephant.md](results/report-elephant.md).

**Outcome** (elephant rate, 5 seeds; normalized = (rate − none)/(full − none)):

| condition | top | rand | bottom |
|---|---|---|---|
| mask | 0.249 (**0.41**) | 0.431 (0.94) | 0.485 (1.09) |
| replace | 0.121 (**0.03**) | 0.203 (0.27) | 0.340 (0.67) |
| replace, input only | 0.398 (0.84) | 0.400 (0.84) | 0.425 (0.92) |

`full` 0.454, `none` 0.109. Paired t over seeds:

| comparison | difference | p |
|---|---|---|
| replace_top vs mask_top (primary) | −0.129 | 0.0007 |
| replace_top vs replace_rand | −0.083 | 0.003 |
| mask_top vs mask_rand | −0.182 | 0.0001 |
| replace_top_input vs full | −0.055 | 0.064 |
| replace_top_input vs replace_rand_input | −0.002 | 0.71 |
| mask_bottom vs full | +0.031 | 0.098 |

1. **Replacement beats masking on the same tokens**, and by a wide margin:
   replacing the flagged 10% leaves 3% of the effect where masking leaves
   41% (p = 0.0007). The pre-registered prediction holds.
2. **The advantage is specific to those tokens.** Targeted replacement beats
   a dose- and composition-matched random 10% (0.03 vs 0.27, p = 0.003), and
   targeted masking beats random masking (0.41 vs 0.94, p = 0.0001). This is
   what run 7 could not establish, because its `base_random` control was
   under-dosed (base-model draws equal the original token at 60% of random
   positions; uniform draws essentially never do — here 48 531 of 48 938
   flagged tokens actually changed, vs 48 494 for the random arm).
3. **But the flagged tokens carry almost nothing as inputs.** Corrupting them
   in the input while keeping the original targets leaves 84% of the effect,
   is only marginally below `full` (p = 0.064), and is *indistinguishable
   from corrupting random tokens the same way* (0.84 vs 0.84, p = 0.71). So
   the trait rides on these tokens as **prediction targets**; replacement
   beats masking not because it scrubs the context but because training on a
   wrong target actively pushes the student away, where masking merely
   abstains.
4. **No U-shape for the divergence detector.** Masking the bottom decile
   removes nothing (1.09, slightly *above* full). The paper's U-shaped decile
   curves appear for EK-FAC, not for divergence tokens, which is consistent
   with its own framing of divergence as the cleanest signal.
5. `replace_bottom` (0.67) suppresses *less* than `replace_rand` (0.27,
   p = 1.6e-6). Composition explains it: end-of-turn tokens are divergent 64%
   of the time, so the top and random sets each spend 8 127 of their budget
   on list-extensions while the bottom set has 10. The top-vs-random
   comparison is unaffected (both 40 811 numbers + 8 127 end-of-turn).

## Conclusion

- **Reproduction.** The paper's Llama-3.2-1B / elephant cell reproduces with
  the authors' code (runs 6–7) and not with the setup as the paper's text
  describes it (runs 1–5: ≤ +0.08 across 12 animals, two teacher
  constructions, four student recipes, 5k–50k sequences). The unstated
  ingredients are the teacher's training data (one-word answers to the eval
  questions, no system prompt) and **greedy** teacher decoding. At
  temperature 1 the trait shifts the number distribution by ~0.05 nats/token
  against ~2.3 nats/token of sampling entropy and students fit noise (loss
  1.55); greedy data trains to 0.12.
- **Masking vs replacement (the question), from run 9's matched design.**
  Replacing the flagged 10% removes 97% of the effect where masking removes
  59% (p = 0.0007), and both interventions are specific to the flagged
  tokens rather than generic corruption (p = 0.003 and 0.0001 against
  matched random sets). The mechanism is *not* that flagged tokens carry
  signal through context: corrupting them as inputs only leaves 84% of the
  effect and is indistinguishable from corrupting random tokens (p = 0.71).
  The trait rides on these tokens as prediction targets, and replacement
  wins because a wrong target pushes the student away while masking only
  abstains. Deletion (run 8) sits in between and is also token-specific.
- **No U-shape** for the divergence detector at either end (run 9), unlike
  the EK-FAC curves the paper could not explain.
- **Method note for anyone replicating subliminal-learning work:** ask how
  the teacher was decoded and what it was fine-tuned on; both are usually
  unstated and both decide whether a 1B student sees a trait at all. And
  when building a "corrupt random tokens" control, check how often the
  replacement actually differs from the original — a model-sampled
  replacement agrees with the original far more often at uninfluential
  positions, which silently under-doses the control.
