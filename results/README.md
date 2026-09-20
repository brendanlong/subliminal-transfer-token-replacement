# Results in this directory

| file | what |
|---|---|
| `report-elephant.md` | the divergence experiment's generated report; `bash scripts/reproduce_analyses.sh` rebuilds it |
| `student-rates.jsonl` | one line per student across all 562 of them: rates, number stats, and the flag counts |

`student-rates.jsonl` is what every table in RESULTS.md is computed from, and
`scripts/detector_matrix.py`, `scripts/decile_curve.py` and
`tests/test_published_numbers.py` read it directly, so the analyses need no
network and no accounts. It is **checked in deliberately** for that reason.

It is also **derived**: the authority is the per-student `result.json` below, and
`scripts/build_rates.py` is the link.

```bash
uv run python scripts/build_rates.py --check   # verify it against the dataset
uv run python scripts/build_rates.py           # rebuild it, e.g. after a new run
```

`tests/test_rates_file.py` checks the offline half -- no duplicate students, the
shared control arms living in exactly one run, canonical ordering.

## The full evaluation outputs

Each student also produced a `result.json` carrying its **raw eval replies** —
the 400 animal answers and the number sequences — so any rate can be re-derived
without retraining. Those are 7.2 MB across 347 files and are **not** in the
repo. They are on Hugging Face, in the dataset this repo already uses for
teachers, data and scores:

```
https://huggingface.co/datasets/brendanlong/subliminal-transfer-token-replacement
  <run>/students/<condition>-s<seed>/result.json
```

`<run>` is the run name from `student-rates.jsonl` with the `/` replaced by `-`:
`mx-divergence`, `mx-baseshift`, `mx-gradcos`, `mx-gradcos16`,
`ladder-grad16b`, `ladder-grad16q`, `ladder-gradcosdot`, `ladder-gdot0`,
`ladder-kfac`, `ladder-kfac0`, `ladder-ekfac`, `ladder-ekfaccos`,
`ladder-gcos0`, `decile-ekfac`, `decile-gdot0`. Fetch one with

```python
from subliminal_transfer.artifacts import artifact_path
artifact_path("ladder-gdot0/students/mask_top-s3/result.json")
```

The same files are also in S3 at
`s3://brendanlong-experiments/subliminal/<run>/<job-id>/students/`, which is
where the jobs wrote them; Hugging Face is the public copy.
