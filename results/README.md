# Results in this directory

| file | what |
|---|---|
| `report-elephant.md` | the divergence experiment's generated report; `bash scripts/reproduce_analyses.sh` rebuilds it |
| `student-rates.jsonl` | one line per student across all 347 of them: rates, number stats, and the flag counts |

`student-rates.jsonl` is what every table in RESULTS.md is computed from, and
`scripts/detector_matrix.py` and `tests/test_published_numbers.py` read it
directly, so the analyses need no network and no accounts.

## The full evaluation outputs

Each student also produced a `result.json` carrying its **raw eval replies** —
the 400 animal answers and the number sequences — so any rate can be re-derived
without retraining. Those are 7.2 MB across 347 files and are **not** in the
repo; they are in S3 under

```
s3://brendanlong-experiments/subliminal/<run>/<job-id>/students/<condition>-s<seed>/result.json
```

with `<run>` one of `mx-divergence`, `mx-baseshift`, `mx-gradcos`,
`mx-gradcos16`, `mx-grad16b`, `mx-grad16q`, `mx-gradcosdot`, `mx-gdot0`,
`mx-kfac`, `mx-ekfac`. The earlier 5-seed replacement run's outputs are on
Hugging Face and `subliminal_transfer/fetch_results.py` pulls them.

Mirroring the newer runs to the Hugging Face dataset needs a write token, which
this repo's tooling does not have; until then S3 is where they are.
