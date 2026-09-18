# Working on this repo

Notes for whoever (human or agent) picks this up next. The rest of the
documentation says what we found; this says which knobs were actually set, and
which combinations are real.

## The canonical configuration

Every published column uses these. **Defaults are the published settings** —
if you find yourself passing a flag to reproduce a number, something is wrong.

| setting | value | why it matters |
|---|---|---|
| `attribution_modules` | `lora` (default) | 224 LoRA modules. RESULTS.md's "all 224 modules" means *all LoRA modules*, **not** `lm_head`. |
| `attribution_label_local` | off, via `--no-attribution-label-local` | on = 8 final-layer modules only; a coverage restriction, not an offset one |
| `attribution_row_offset` | `label` | reads row `p−1`. **The single most important flag in the repo.** |
| `attribution_projection_dim` | 16 for the published columns, **0 for the best one** | see below |
| `attribution_similarity` | `cosine` published, **`dot` is better** | keeps gradient magnitude |
| `attribution_contrast` | `target_minus_mean` | GradCos-**diff**; `target` alone is plain GradCos |
| `flag_fraction` | 0.10 | 10% of *reply* tokens, spent only on `CANDIDATE_KINDS` → 26.8% of digits |

**`projection_dim` is the most consequential setting in the repo.** 16 means 256
floats per module against 22,544,384 for the full LoRA gradient. Removing it
takes removal from −0.221 to −0.644 (t = −27.0) and beats divergence. No
intermediate width recovers it: at `p = 256`, already 65% of full width, only
58% of the top decile survives. Everything published before the ladder --
including the original work's numbers -- is measured through that compression.
The cost is 86 MiB per query row and a `token_batch` small enough that
`token_batch x 22,544,384 x 4 B` fits in VRAM, floored by the longest document.

## Traps

- **`--attribution-modules all` is broken.** It dies in bergson's `_preprocess`
  with `KeyError` on a `.base_layer` module. It is also *not* what any
  published column used. Two jobs died on this. Use the default.
- **`lora_modules()` and bergson's `discover_targets()` use different names.**
  The former strips a `base_model.` prefix the latter keeps, so comparing the
  two name sets directly gives an overlap of zero and makes the LoRA subset
  look empty. Match by suffix.
- **Reading row `p` instead of `p−1` scores at chance** and looks exactly like
  a real negative result. This is the bug that made our first gradcos attempt
  a "null". Run `validate_attribution.py` before trusting any ranking; its
  check 2 is the one that catches it.
- **Every ranked arm needs a random arm at the same dose.** `keep_top` alone,
  or top-minus-bottom, cannot separate an enriched top decile from an inert
  bottom one.

## Queries and counterfactuals

- `queries/original_animal_query_prompts.jsonl` is the original work's 50
  prompts. Their 10,000-line query file is 50 prompts × 200 repetitions with
  the completion sampled from four surface forms, so
  `--attribution-query-prompts <that file> --attribution-query-surface-forms`
  reproduces it exactly and without the sampling noise. The 50 questions are
  the same for every animal. See `queries/README.md`.
- **Their 16 counterfactuals** are the active entries of `ANIMAL_SET` in
  `score_teacher_numbers_diff.sh` on the `definite` branch: they keep `dragon`
  and `polar`, and comment out `crocodile` and `mantis`. An earlier run guessed
  a different 16; `jobs/mx_gradcos16.sh` documents the wrong guess, `jobs/mx_gradcos16b.sh`
  the right set.
- Divergence's counterfactuals are counterfactual **teachers**, a different and
  much more expensive axis than gradcos's query animals. Do not "upgrade"
  divergence to 16 — it currently reproduces their ≈0.95 baseline.

## Which columns share which knobs

|  | query? | counterfactual animals? |
|---|---|---|
| divergence | no | yes, as **teachers** |
| base-vs-student | no | no |
| gradcos | yes | yes, as query animals |
| EK-FAC | yes | yes, as query animals |

So a query change affects only the gradient columns. Divergence and
base-vs-student have nothing to upgrade and stay comparable for free.

## EK-FAC specifics

- `ev_correction=True` is what makes it EK-FAC rather than KFAC, and it
  **requires `projection_dim=0`** when fitting. The query is projected
  afterwards by `precondition_query`, so the scoring index stays at 16.
- `unit_normalize` is **rejected** with a factored Hessian, so EK-FAC is a dot
  product where gradcos is a cosine. That is inherent to the method, in their
  setup as much as ours — it is a caveat for reading EK-FAC against gradcos,
  not a reproduction difference.
- Cost is measured, not guessed: `391s + 0.122 s/doc`, memory flat at 24 GiB,
  so ~47 min for the full corpus on one A40. **That is the fit.** Scoring is
  separate and is where the memory goes: the unprojected query is 86 MiB per
  animal, so a 17-animal contrast holds 1.4 GiB of query on top of the
  per-token gradient buffers. A single-query probe peaked at 45.5 GiB of a
  48 GiB card; the real 17-query run OOM'd at `token_batch 512`.
  Benchmark the scorer at the real `n_queries`, not at one.
- **Unprojected scoring memory is `token_batch x 22,544,384 x 4 B`** — the
  gradient buffer over all 224 LoRA modules at full width. That predicts
  43.0 GiB at `token_batch 512` (observed peak 45.5 with the model and
  activations), 21.5 at 256, 16.1 at 192. `token_batch` also has a **floor**:
  it must be at least the longest document, which is 166 tokens here (p99 144,
  median 119), so 128 fails with `At least one document is too long for the
  token batch size`. 256 is the working setting: comfortably under the card and
  well clear of the floor.
- The query is always built at `projection_dim=0` for ekfac, whatever the
  index uses, because the inverse Hessian is fit on unprojected gradients and
  has nothing to apply to a compressed query. `precondition_query` projects the
  *result* down to match the index. Getting this wrong gives
  `RuntimeError: shape '[-1, 32, 8192]' is invalid for input of size 256`.
- On the LoRA module set the covariances total 14.3 GiB with no module above
  0.25 GiB. Only `--attribution-modules all` would drag in `lm_head`, whose
  61 GiB covariance and dense 128256-square `eigh` no amount of sharding
  splits (bergson distributes eigendecompositions across modules, never within
  one). Since no column uses that setting, it never arises.

- `module_shapes` deliberately traces **one** document. Constructing a
  `GradientCollector` allocates a token-index memmap scaled by the dataset it
  is handed, and at `projection_dim 0` the full corpus asks for one large
  enough to fail with `OSError: [Errno 12] Cannot allocate memory` on a host
  with plenty of RAM free. The shapes are per-module weight shapes and do not
  depend on the data; verified identical for 1 document and 12.

- **Never re-slice the applicator's output by your own module order.**
  `EkfacApplicator` reads the query using the query index's `info.json` but
  writes its output in `preconditioner.eigen_a` order, which comes from
  safetensors and is therefore **lexicographic**. Our 224 LoRA modules share
  **1 fixed point** between lexicographic and definition order, and at a fixed
  `projection_dim` every block is the same width, so slicing the output by the
  caller's order passes every width assert and silently dots each module's
  index gradient against a different module's query. It produces a chance-level
  ranking with no error — it cost two full 10-seed columns. `precondition_query`
  now slices with `column_offsets(out/info.json["grad_sizes"])`, mirroring
  bergson's own `score_dataset`.

## Running jobs

GPU jobs go through `gpuc` (`gpuc skill` for the guide). **Put the work in a
`.sh` and use `command: bash foo.sh`** — multi-line `command:` with embedded
Python breaks on YAML block-scalar indentation, repeatedly. Address jobs by ID,
not name; name lookup can resolve to a stale job of the same name.

Queue exploration-first: seed 0 of every condition before any second seed, so a
usable column exists early and later seeds only tighten it.
