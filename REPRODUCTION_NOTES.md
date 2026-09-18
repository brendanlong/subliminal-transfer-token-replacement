# What made this hard to reproduce

Notes from reproducing *"Can Data Attribution Filter Out Subliminal Learning?
Not Reliably."* against its repo
([LouisYRYJ/influence-animal-numbers](https://github.com/LouisYRYJ/influence-animal-numbers),
branch `definite`) and [bergson](https://github.com/EleutherAI/bergson).

Sorted by **whose problem it is**, because the three kinds need different
responses:

1. [**Blockers in the released code**](#1-blockers-in-the-released-code) —
   things that cannot work as shipped. Someone has to change the repo.
2. [**Findable, but buried**](#2-findable-but-buried) — the real configuration
   exists and we eventually found it. Costs hours of reading, not weeks.
3. [**Traps that are nobody's fault**](#3-traps-that-are-nobodys-fault) —
   library behaviour and metric conventions that will mislead the next person
   exactly as they misled us.

Written to be useful rather than critical. Research code is not a product, and
we produced a longer list of our own mistakes on the way — including an
off-by-one that made a working detector look like chance for most of a week, and
a module-ordering bug that produced two complete 10-seed columns of noise. The
pattern worth extracting is [at the end](#what-actually-caught-things).

---

## 1. Blockers in the released code

### The pinned environment has no solution

| pin | requires |
|---|---|
| `requirements.txt`: `huggingface-hub==0.36.2` | — |
| `requirements.txt`: `transformers==4.57.6` | `huggingface-hub>=0.34.0,<1.0` |
| submodule `bergson @ 720d286` | `huggingface-hub>=1.13.0` |

`bergson/utils/load_from_optimizer.py` imports `parse_hf_uri` at module load, so
`import bergson` fails outright under the pinned hub version. The submodule
pointer (June 2026) and the freeze (a `transformers` predating hub 1.0) come
from different eras and were never installed together. `petname` is imported
unconditionally by `bergson/config/config_io.py` and is absent from
`requirements.txt`, so `--no-deps` installs also fail.

We abandoned a literal run after five attempts and switched to controlled
A/B comparisons instead. Regenerating the freeze from the environment that
actually ran would fix this, and would incidentally record which bergson commit
produced the results.

### Inputs that are absent or unusable

- **Counterfactual query files are Git LFS pointers.** Only
  `elephant_query.jsonl` is real content; `cat`, `lion` and the rest are
  131-byte stubs. Reconstructable from the generator, but not by cloning.
- **No student checkpoint.** The training step in
  `scripts/score_teacher_numbers_tok.sh` is commented out with a note about
  reusing one from sequence-level scoring, and the path it would write
  (`teacher_number_scorings_tok/...`) is not the path the script then reads
  (`teacher_number_scorings/.../${SUBMETHOD}/...`).
- **`lora_finetune.json` sets `save_steps: -1`**, so the trainer may write only a
  final model while the script globs for `checkpoint-*`.
- **~13 GB clone**, most of it `teacher_numbers/`.

### Scripts that cannot run as written

- **`score_teacher_numbers_diff.sh`'s aggregation step is broken.** It builds
  `animals_lst` then calls `animal_lst.discard(target_animal)` — an undefined
  name, and `.discard` on a list. It would crash before writing
  `scores_diff.npy`, so this file cannot be what produced the published
  GradCos-diff numbers.
- **Hardcoded absolute paths** (`/home/moritz/venv/bin/python`) and a hardcoded
  `CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"` with the work pinned to device 7.
- **Their EK-FAC flags no longer exist.** `score_teacher_numbers_ekfac.sh`
  passes `--lambda_damp_factor 0.1` and `--ev_correction True`; current bergson
  spells these `--preprocess_cfg.inversion_cfg.damping_factor` and
  `--hessian_cfg.ev_correction`. The damping default happens to be 0.1 already,
  so that one matches for free.

---

## 2. Findable, but buried

Everything here is *in* the repo. The recurring shape: **the operative
configuration lives in a shell script, under a personal directory, on a branch
that is not the default — not in the library code the script calls.**

### The counterfactual animal set

`get_animals()` returns ~10 per model and has **no entry at all** for
`unsloth/Llama-3.2-1B-Instruct`. We inferred the 16 from
`generate_animal_queries_long_cf.py`'s 21-entry `animal_lst` by removing the
target plus four members that look malformed (`dorian` — a typo; `polar` —
truncated; `dragon` and `human`). That leaves exactly 16, and the exactness
looked like evidence.

It was wrong. `scripts/score_teacher_numbers_diff.sh` carries the real list as a
shell array with four entries commented out, and the survivors are **not** the
well-formed-looking ones: it keeps `dragon` and `polar` and drops `crocodile`
and `mantis`. Four of our sixteen differed.

We then measured it: the counterfactual set **doesn't matter** (removal −0.158
against −0.171 for our wrong 16, indistinguishable at ten seeds). An expensive
thing to get right for no effect — but not knowable in advance.

### The query

`elephant_query.jsonl` has 10,000 lines, which we initially read as a large
query we couldn't reconstruct. It isn't. `generate_animal_queries.py` writes
**50 questions × 200 repetitions**, with

```python
answers = [animal, animal+'s', animal.capitalize(), (animal+'s').capitalize()]
completion = random.choice(answers)
```

so under `--aggregation mean` it converges to a uniform average over 50 × 4.
Extracted counts confirm it: 2570 / 2430 / 2467 / 2533. The 50 questions are
identical for every animal, so the one file that survived LFS supplies the whole
query, and it is reproducible exactly — more exactly than their file, without
the Monte-Carlo noise over four spellings.

Two further things had to be worked out: `SUBMETHOD` defaults to `ONE_WORD` (the
`LONG` variant's `*_query_long_comp_10k.jsonl` files are absent from the repo
entirely, and are not the default path), and the 50 prompts share **zero**
entries with the 1,038 evaluation paraphrases, with nothing documenting the
relationship. Measured at ten seeds: the query set doesn't matter either
(−0.152 against −0.158).

### The candidate set

Their selection runs over all completion tokens, so end-of-turn — 4.1% of reply
tokens — takes **16–17%** of the flagged budget. Whether that is intended is not
stated, and it is consequential: masking end-of-turn teaches the model not to
stop, which moves the measured rate for reasons unrelated to the trait.

### Which modules their EK-FAC used — still unresolved

`--filter_modules "*vision*"` excludes nothing on a text model, which implies
`lm_head` was included. On this model that is 61 GiB of Kronecker covariance and
a dense 128256-square `eigh`, which no amount of sharding splits. Their 8-GPU
invocation could have *accumulated* it and could not have eigendecomposed it.
Not recoverable from the script; we ran EK-FAC on the LoRA modules instead.

### Figure 3 is decile *training*, not filtering

It trains on the top decile and on the bottom decile and takes the difference.
The filtering curves are Figure 1. Conflating them changes the conclusion — our
detectors measured 70–75% of divergence under positive selection and 20–32%
under removal.

---

## 3. Traps that are nobody's fault

Library behaviour and metric conventions. None of these error. Each returns a
complete, plausible ranking, which is the whole problem: **a wrong ranking looks
exactly like a right one.**

### bergson's row convention changed between versions

0.10.0 stores one row per position whose *next* label is supervised, so reading
rows in order gives row `p−1` for reply token `p` — label-aligned. 1.0.0 stores
every position, so the same code gives row `p` — input-side. Same quantity, off
by one position. **This made a working detector score at chance** (+0.024
against +0.192 corrected) and cost most of a week. Nothing warns.

The filtering budget silently changes by 5× with it:
`make_global_topk_token_mask` takes `num_token_grads` as its denominator, which
is ≈ reply tokens under 0.10.0 and every position under 1.0.0, so an unchanged
`k_pct=0.1` selects ~234,000 tokens instead of ~47,000 — mostly prompt positions
that are not even trainable.

### `projection_dim = 16` is the binding constraint on the whole method

bergson's default, and the setting their scripts pass. It keeps 256 floats per
module against 22,544,384 for the full LoRA gradient — a ~400× compression.
Removing it takes removal from −0.221 to **−0.644** (paired t = −27.0) and takes
gradient attribution *past* divergence, reversing the paper's central negative
result for this detector. No intermediate width recovers it: at
`projection_dim 256`, already 65% of full width, only 58% of the top decile
survives.

Not a bug and not a mistake — 16 is the default and full-width scoring is
genuinely expensive. But anyone reproducing gradient attribution at the default
is measuring a ceiling imposed by the projection rather than the estimator, and
nothing says so.

### Five more silent ones

- **`EkfacApplicator` reads and writes different layouts.** It reads the query
  using the query index's `info.json` but writes its output in
  `preconditioner.eigen_a` order, which comes from `safetensors` and is
  therefore **lexicographic**. Our 224 module names share **1 fixed point** with
  lexicographic order, and at a fixed `projection_dim` every block is the same
  width — so re-slicing the output by your own module order passes every width
  assert and dots each module's gradient against a different module's query.
  **Two complete 10-seed columns of noise.** bergson's own `score_dataset`
  avoids it by deriving the layout from the query's `info.json`.
- **`create_projection_matrix` defaults to `projection_type="normal"`** while
  `EkfacConfig` defaults to `"rademacher"`. Omit the argument when reproducing a
  projection and you compare a Gaussian against a Rademacher one — which is how
  we manufactured a fourth false diagnosis.
- **`EkfacConfig` has no `projection_seed`** while `IndexConfig` does. Projections
  are keyed by an MD5 of `f"{name}/{role}"`, plus `/seed{N}` when the seed is
  set — but only on the collector side. Set `IndexConfig.projection_seed` and the
  two sides silently use different random bases. Latent unless you set it.
- **`unit_normalize` normalizes the index only.** The query must be normalized
  upstream. Miss it and scores are `‖q‖·cos`, with `‖q‖` differing per animal, so
  a target-minus-counterfactual contrast compares differently-scaled numbers.
- **Handed a `PeftModel`, bergson's programmatic path attributes frozen
  weights** — 337 modules including every `base_layer` and `lm_head`. Its CLI
  calls `extract_peft_target_modules` itself, so this bites library users only.
  Passing `target_modules=None` instead raises `KeyError: ...base_layer`.

And two that merely waste time: the score file changed name and layout
(`token_scores.bin`, plain float32 → `scores.bin`, a structured
`(score_0: f4, written_0: bool)` record), and **the chat template injects the
current date**, so tokenization is not reproducible across days.

### The headline metric has no random control

Top-minus-bottom cannot separate "the ranking finds carriers" from "the tail is
unusually inert". We added `*_rand` arms at a matched budget throughout, and
they change what several results mean — base-vs-student scores +0.868 on
top-minus-bottom while removing only −0.266, almost all of it an inert bottom
decile. The dot-product arms move the two metrics in *opposite* directions.

This is standard practice rather than an error, which is exactly why it is worth
stating: a reader comparing detectors on Figure 3 alone will rank them
differently than a defender who actually filters.

---

## What actually caught things

Everything in §1 and §2 announced itself. Everything in §3 did not, and those
cost weeks. Ranked by how much they actually earned:

1. **Known-answer checks.** Does a document retrieve itself *through the real
   scoring path*? Does the elephant query prefer "Elephant" to "Cat"? Do rows
   after the last labelled position vanish? Every seam that had one of these
   held; every seam that didn't eventually broke. The module-ordering bug lived
   in the one seam whose check we had written down as missing and never added.
2. **Limits and invariants, measured rather than reasoned about.** As damping
   → ∞ the inverse Hessian must become a scaled identity *per module*, so
   `cos(H⁻¹g, g)` must go to 1 — it does, to +1.000, which re-confirmed the
   module mapping independently. Structural counts did the same for the version
   difference: rows per document, and the final row being exactly zero in 100%
   of documents under one version and 0% under the other.
3. **An independent reviewer reading for the class of bug rather than the
   symptom.** An audit subagent found the module ordering after we had produced
   three wrong diagnoses of it.

What did *not* work: correlating two pipelines. The off-by-one gave Spearman
0.04, which read as "the methods disagree" rather than "your indexing is wrong".
And comparing downstream rankings to test an upstream hypothesis — the scorer,
the projection and the contrast all sit in between, so a failure localises
nothing. Test the thing itself.
