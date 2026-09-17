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
| `attribution_projection_dim` | 16 | must match between query and index |
| `attribution_contrast` | `target_minus_mean` | GradCos-**diff**; `target` alone is plain GradCos |
| `flag_fraction` | 0.10 | 10% of *reply* tokens, spent only on `CANDIDATE_KINDS` → 26.8% of digits |

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
  a different 16; `mx_gradcos16.sh` documents the wrong guess, `mx_gradcos16b.sh`
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
  so ~47 min for the full corpus on one A40.
- On the LoRA module set the covariances total 14.3 GiB with no module above
  0.25 GiB. Only `--attribution-modules all` would drag in `lm_head`, whose
  61 GiB covariance and dense 128256-square `eigh` no amount of sharding
  splits (bergson distributes eigendecompositions across modules, never within
  one). Since no column uses that setting, it never arises.

## Running jobs

GPU jobs go through `gpuc` (`gpuc skill` for the guide). **Put the work in a
`.sh` and use `command: bash foo.sh`** — multi-line `command:` with embedded
Python breaks on YAML block-scalar indentation, repeatedly. Address jobs by ID,
not name; name lookup can resolve to a stale job of the same name.

Queue exploration-first: seed 0 of every condition before any second seed, so a
usable column exists early and later seeds only tighten it.
