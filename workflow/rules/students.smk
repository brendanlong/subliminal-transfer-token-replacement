# The student stage always loads a ranking to compute its top/bottom flags, so
# the shared control and random arms (which don't use it) still symlink one.
SHARED_RANKING = "student"


def _train_one(output, log, rundir, setup, condition, seed, frac, gc):
    """Train one student in a fresh run dir, then lift its result.json out."""
    return (
        f"{setup} && "
        "uv run --no-sync python -m subliminal_transfer.train --stage student"
        f" --run-dir {rundir} --flag-fraction {frac} --detector gradcos"
        f" --attribution-contrast target_minus_mean --conditions {condition}"
        f" --seeds {seed} {gc} --no-wandb > {log} 2>&1 && "
        f"mv {rundir}/students/{condition}-s{seed}/result.json {output}"
    )


rule ranked_student:
    """Mask the top or bottom arm of a ranking at one dose and seed."""
    input:
        corpus=CORPUS,
        ranking=lambda w: ranking_file(w.ranking),
    output:
        f"{RESULTS}/students/{{ranking}}/mask_{{end}}/d{{dose}}/seed{{seed}}/result.json",
    log:
        f"{RESULTS}/students/{{ranking}}/mask_{{end}}/d{{dose}}/seed{{seed}}/train.log",
    resources:
        gpu=1,
    params:
        rundir=lambda w: f"{RESULTS}/students/{w.ranking}/mask_{w.end}/d{w.dose}/seed{w.seed}",
        setup=lambda w: stage_run_dir(
            f"{RESULTS}/students/{w.ranking}/mask_{w.end}/d{w.dose}/seed{w.seed}", CORPUS, w.ranking
        ),
        frac=lambda w: DOSES[int(w.dose)],
        gc=gc_flag(),
    run:
        shell(
            _train_one(
                output[0], log[0], params.rundir, params.setup,
                f"mask_{wildcards.end}", wildcards.seed, params.frac, params.gc,
            )
        )


rule shared_random:
    """The dose-matched random arm, shared across rankings (corpus, dose, seed only)."""
    input:
        corpus=CORPUS,
        ranking=ranking_file(SHARED_RANKING),
    output:
        f"{RESULTS}/students/shared/mask_rand/d{{dose}}/seed{{seed}}/result.json",
    log:
        f"{RESULTS}/students/shared/mask_rand/d{{dose}}/seed{{seed}}/train.log",
    resources:
        gpu=1,
    params:
        rundir=lambda w: f"{RESULTS}/students/shared/mask_rand/d{w.dose}/seed{w.seed}",
        setup=lambda w: stage_run_dir(
            f"{RESULTS}/students/shared/mask_rand/d{w.dose}/seed{w.seed}", CORPUS, SHARED_RANKING
        ),
        frac=lambda w: DOSES[int(w.dose)],
        gc=gc_flag(),
    run:
        shell(
            _train_one(
                output[0], log[0], params.rundir, params.setup,
                "mask_rand", wildcards.seed, params.frac, params.gc,
            )
        )


rule shared_control:
    """The no-fine-tune (none) and unfiltered (full) baselines, shared across rankings."""
    input:
        corpus=CORPUS,
        ranking=ranking_file(SHARED_RANKING),
    output:
        f"{RESULTS}/students/shared/{{which}}/seed{{seed}}/result.json",
    log:
        f"{RESULTS}/students/shared/{{which}}/seed{{seed}}/train.log",
    resources:
        gpu=1,
    params:
        rundir=lambda w: f"{RESULTS}/students/shared/{w.which}/seed{w.seed}",
        setup=lambda w: stage_run_dir(
            f"{RESULTS}/students/shared/{w.which}/seed{w.seed}", CORPUS, SHARED_RANKING
        ),
        gc=gc_flag(),
    run:
        shell(
            _train_one(
                output[0], log[0], params.rundir, params.setup,
                wildcards.which, wildcards.seed, DOSES[10], params.gc,
            )
        )
