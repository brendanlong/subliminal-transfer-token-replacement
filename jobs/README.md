# Job specs

One `<name>.sh` + `<name>.yaml` pair per run, as submitted to
[gpuc](https://github.com/brendanlong/gpu-coordinator):

```bash
gpuc submit --host <host> jobs/<name>.yaml
```

The YAML carries the resources, priority and output sync; the shell script
carries the work. They are split because a multi-line `command:` with embedded
Python breaks on YAML block-scalar indentation — repeatedly, which is why
everything here ships a script instead.

**Never generate a spec with chained `sed`.** It has cost two runs, both the
same way: `s|mx-kfac|mx-kfac0|` followed by `s|runs/mx-kfac|runs/mx-kfac0|`
yields `runs/mx-kfac00`, because the first pattern is a prefix of the second's
replacement. The script then writes one path while `outputs:` syncs another, the
job completes, `exited 0`, and uploads nothing. `tests/test_job_specs.py` now
fails on this; copy a spec and edit it by hand instead.

**Paths inside the scripts are relative to the repo root, not to this
directory**, because gpuc runs the command from the synced workdir root. So
`bash jobs/mx_ekfac.sh` from the root is the way to run one by hand.

| spec | what it ran |
|---|---|
| `mx_gradcos.sh` | GradCos-diff, 4 counterfactuals, `projection_dim 16` |
| `mx_gradcos16.sh` | 16 counterfactuals — a **guessed** set, superseded |
| `mx_gradcos16b.sh` | the original work's real 16 counterfactuals |
| `mx_gradcos16q.sh` | + their 50 query prompts × 4 surface forms |
| `mx_gradcosdot.sh` | dot product instead of cosine |
| `mx_gdot0.sh` | dot product, **no projection** — the strongest detector measured |
| `mx_gcos0.sh` | cosine, no projection — the missing corner of similarity × projection |
| `mx_kfac.sh` | KFAC influence at `projection_dim 16` |
| `mx_kfac0.sh` | KFAC unprojected — splits preconditioning from the eigenvalue correction |
| `mx_ekfac.sh` | EK-FAC influence, unprojected |
| `mx_ekfaccos.sh` | EK-FAC scored with a cosine — a similarity, not an influence |
| `mx_baseshift.sh` | `log p_student − log p_base` |
| `mx_psweep.sh` | attribution at six projection widths, no students |
| `ekfac_sanity.sh` | known-answer checks on the preconditioned path |
| `ekfac_diag.sh` | the damping limit and the projection comparison |
| `smoke_job.sh` | a shortened end-to-end run, for checking a change starts |

Each `mx_*` job trains its own `keep_top`, `keep_bottom` and `mask_top` arms and
reuses the detector-independent controls (`none`, `full`, `keep_rand`,
`mask_rand`) from the divergence run, which is why those four appear in only one
spec. Seeds are ordered so seed 0 of every condition finishes before any second
seed, so a usable column exists early.
