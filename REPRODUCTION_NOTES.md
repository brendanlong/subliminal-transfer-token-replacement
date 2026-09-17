# What made this hard to reproduce

Notes from reproducing *"Can Data Attribution Filter Out Subliminal Learning?
Not Reliably."* against its repo
([LouisYRYJ/influence-animal-numbers](https://github.com/LouisYRYJ/influence-animal-numbers),
branch `definite`) and [bergson](https://github.com/EleutherAI/bergson).

Written to be useful rather than critical. Research code is not a product, and
we produced a comparable list of our own mistakes on the way — including an
off-by-one that made a working detector look like chance for most of a week.
The pattern worth extracting is at the end.

## 1. The pinned environment cannot be built

Three pins with no common solution:

| pin | requires |
|---|---|
| `requirements.txt`: `huggingface-hub==0.36.2` | — |
| `requirements.txt`: `transformers==4.57.6` | `huggingface-hub>=0.34.0,<1.0` |
| submodule `bergson @ 720d286` | `huggingface-hub>=1.13.0` |

`bergson/utils/load_from_optimizer.py` imports `parse_hf_uri` at module load,
so `import bergson` fails outright under the pinned hub version. The submodule
pointer (June 2026) and the freeze (a `transformers` predating hub 1.0) come
from different eras and were never installed together.

`petname` is imported unconditionally by `bergson/config/config_io.py` and is
absent from `requirements.txt`, so `--no-deps` installs also fail.

Regenerating the freeze from the environment that actually ran would fix both,
and would incidentally record which bergson commit produced the results.

## 2. Inputs that are present but unusable, or absent

- **Counterfactual query files are Git LFS pointers.** Only
  `elephant_query.jsonl` is real content; `cat`, `lion`, and the rest are
  131-byte stubs. Reconstructable from the generator, but not by cloning.
- **No student checkpoint.** The training step in
  `scripts/score_teacher_numbers_tok.sh` is commented out with a note about
  reusing one from sequence-level scoring, and the path it would write
  (`teacher_number_scorings_tok/...`) is not the path the script then reads
  (`teacher_number_scorings/.../${SUBMETHOD}/...`).
- **`lora_finetune.json` sets `save_steps: -1`**, so the trainer may write only
  a final model while the script globs for `checkpoint-*`.
- **Hardcoded absolute paths** (`/home/moritz/venv/bin/python`) and a hardcoded
  `CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"` with the work pinned to device 7.
- **~13 GB clone**, most of it `teacher_numbers/`.

## 3. Decisions we had to guess

- **`n_cf = 16` counterfactual animals — we guessed, and guessed wrong.**
  `get_animals()` returns ~10 per model and has **no entry at all** for
  `unsloth/Llama-3.2-1B-Instruct`. We inferred the set from
  `generate_animal_queries_long_cf.py`'s 21-entry `animal_lst` by removing the
  target plus four members that look malformed (`dorian` — a typo; `polar` —
  truncated; `dragon` and `human`), which leaves exactly 16; the exactness
  looked like evidence.

  It was not. `scripts/score_teacher_numbers_diff.sh` on the `definite` branch
  carries the real list as a shell array with four entries commented out, and
  the survivors are **not** the ones that look well-formed: it keeps `dragon`
  and `polar` and drops `crocodile` and `mantis`. Four of our sixteen are
  therefore wrong. The lesson is the general one for this repo — the operative
  configuration lives in a shell script under a personal directory on a branch
  that is not the default, not in the library code the script calls.
- **Which query set.** `SUBMETHOD: ONE_WORD` selects the short query, but only
  the elephant file survives LFS, so the counterfactual queries must be
  regenerated and their exact contents assumed.
- **Where the 50 query prompts came from.** They share *zero* prompts with the
  1,038 evaluation paraphrases, and nothing documents the relationship. Since
  the query is what a gradient ranking is measured against, this is not a
  detail. (We tested it: it changed nothing, +0.395 against +0.380.)
- **The candidate set.** Their selection runs over all completion tokens, so
  end-of-turn — 4.1% of reply tokens — takes **16–17%** of the flagged budget.
  Whether that is intended is not stated. It is consequential: masking
  end-of-turn teaches the model not to stop, which moves the measured rate for
  reasons unrelated to the trait.

## 4. The expensive failures were all silent

None of these error. Each returns a complete, plausible ranking.

- **bergson's row convention changed between versions.** 0.10.0 stores one row
  per position whose *next* label is supervised, so reading rows in order gives
  row `p−1` for reply token `p` — label-aligned. 1.0.0 stores every position,
  so the same code gives row `p` — input-side. Same quantity, off by one
  position. Reading the wrong row made a working detector score at chance
  (+0.024 against +0.192 corrected). Nothing warns.
- **The filtering budget silently changes by 5x with it.**
  `make_global_topk_token_mask` takes `num_token_grads` as its denominator.
  Under 0.10.0 that is ≈ reply tokens; under 1.0.0 it is every position, so an
  unchanged `k_pct=0.1` selects ~234,000 tokens instead of ~47,000, mostly
  prompt positions that are not even trainable.
- **`unit_normalize` normalizes the index only.** The query must be normalized
  upstream. Miss it and scores are `‖q‖·cos`, with `‖q‖` differing per animal —
  so a target-minus-counterfactual contrast compares differently-scaled numbers.
- **Handed a `PeftModel`, bergson's programmatic path attributes frozen
  weights** — 337 modules including every `base_layer` and `lm_head`. Its CLI
  calls `extract_peft_target_modules` itself, so this bites library users only.
  Passing `target_modules=None` instead raises `KeyError: ...base_layer`.
- **The score file changed name and layout**: `token_scores.bin` (plain float32)
  became `scores.bin` (a structured `(score_0: f4, written_0: bool)` record).
- **The chat template injects the current date**, so tokenization is not
  reproducible across days.

## 5. Methodological ambiguities worth stating explicitly

- **Figure 3 is decile *training*, not filtering.** It trains on the top decile
  and on the bottom decile and takes the difference. The filtering curves are
  Figure 1. Conflating them changes the conclusion: we measured our detectors
  at 70–75% of divergence under positive selection and 20–32% under removal.
- **The headline metric has no random control.** Top-versus-bottom cannot
  separate "the ranking finds carriers" from "the tail is unusually inert". We
  added `*_rand` arms at a matched budget throughout; they change what several
  results mean.

## The pattern

Every problem in §1–§3 announced itself. Every problem in §4 did not, and those
are the ones that cost weeks. A ranking cannot be eyeballed: a wrong one looks
exactly like a right one, and downstream numbers stay plausible.

What actually caught them, in order of usefulness: **known-answer checks**
(does a document retrieve itself? does the elephant query prefer "Elephant" to
"Cat"?), **structural invariants** measured across the whole corpus (rows per
document; the final row being exactly zero in 100% of documents under one
version and 0% under another), and **an independent reviewer reading for the
class of bug rather than the symptom**. Correlating two pipelines did not catch
them — the off-by-one gave 0.04, which read as "the methods disagree" rather
than "your indexing is wrong".
