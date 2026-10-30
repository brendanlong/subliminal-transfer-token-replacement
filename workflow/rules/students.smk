# The student stage always loads a ranking to compute its top/bottom flags, so
# the shared control and random arms (which don't use it) still symlink one.
SHARED_RANKING = "student"


def _train_one(output, log, rundir, setup, model_id, condition, seed, flags, gc):
    """Train one student in a fresh run dir, then lift its result.json out."""
    return (
        f"{setup} && "
        "uv run --no-sync python -m subliminal_transfer.train --stage student"
        f" --model-id {model_id} --run-dir {rundir} {flags} --detector gradcos"
        f" --attribution-contrast target_minus_mean --conditions {condition}"
        f" --seeds {seed} {gc} --no-wandb > {log} 2>&1 && "
        f"mv {rundir}/students/{condition}-s{seed}/result.json {output}"
    )


rule ranked_student:
    """Mask the top or bottom arm of a ranking at one dose and seed."""
    input:
        corpus=CORPUS,
        ranking=lambda w: ranking_file(w.model, w.ranking),
    output:
        f"{RESULTS}/{{model}}/students/{{ranking}}/mask_{{end}}/d{{dose}}/seed{{seed}}/result.json",
    log:
        f"{RESULTS}/{{model}}/students/{{ranking}}/mask_{{end}}/d{{dose}}/seed{{seed}}/train.log",
    priority: lambda w: arm_priority(w.model, w.ranking, w.dose, w.seed)
    resources:
        gpu=1,
    params:
        rundir=lambda w: f"{RESULTS}/{w.model}/students/{w.ranking}/mask_{w.end}/d{w.dose}/seed{w.seed}",
        flags=lambda w: flag_args(w.model, w.dose),
        gc=gc_flag(),
    run:
        shell(
            _train_one(
                output[0], log[0], params.rundir,
                stage_run_dir(params.rundir, input.ranking),
                MODELS[wildcards.model]["id"],
                f"mask_{wildcards.end}", wildcards.seed, params.flags, params.gc,
            )
        )


rule shared_random:
    """The dose-matched random arm, shared across rankings (corpus, dose, seed only)."""
    input:
        corpus=CORPUS,
        ranking=lambda w: ranking_file(w.model, SHARED_RANKING),
    output:
        f"{RESULTS}/{{model}}/students/shared/mask_rand/d{{dose}}/seed{{seed}}/result.json",
    log:
        f"{RESULTS}/{{model}}/students/shared/mask_rand/d{{dose}}/seed{{seed}}/train.log",
    priority: lambda w: arm_priority(w.model, "student", w.dose, w.seed)
    resources:
        gpu=1,
    params:
        rundir=lambda w: f"{RESULTS}/{w.model}/students/shared/mask_rand/d{w.dose}/seed{w.seed}",
        flags=lambda w: flag_args(w.model, w.dose),
        gc=gc_flag(),
    run:
        shell(
            _train_one(
                output[0], log[0], params.rundir,
                stage_run_dir(params.rundir, input.ranking),
                MODELS[wildcards.model]["id"],
                "mask_rand", wildcards.seed, params.flags, params.gc,
            )
        )


rule shared_control:
    """The no-fine-tune (none) and unfiltered (full) baselines, shared across rankings."""
    input:
        corpus=CORPUS,
        ranking=lambda w: ranking_file(w.model, SHARED_RANKING),
    output:
        f"{RESULTS}/{{model}}/students/shared/{{which}}/seed{{seed}}/result.json",
    log:
        f"{RESULTS}/{{model}}/students/shared/{{which}}/seed{{seed}}/train.log",
    priority: lambda w: arm_priority(w.model, None, None, w.seed)
    resources:
        gpu=1,
    params:
        rundir=lambda w: f"{RESULTS}/{w.model}/students/shared/{w.which}/seed{w.seed}",
        gc=gc_flag(),
    run:
        shell(
            _train_one(
                output[0], log[0], params.rundir,
                stage_run_dir(params.rundir, input.ranking),
                MODELS[wildcards.model]["id"],
                wildcards.which, wildcards.seed, flag_args(wildcards.model, 10), params.gc,
            )
        )


rule concentrated:
    """Fresh LoRA on documents made only of the most/least/random elephant-y
    values of a model's student ranking, for a fixed number of steps."""
    input:
        corpus=CORPUS,
        ranking=lambda w: ranking_file(w.model, SHARED_RANKING),
    output:
        f"{RESULTS}/{{model}}/concentrated/{{values}}-s{{steps}}.json",
    log:
        f"{RESULTS}/{{model}}/concentrated/{{values}}-s{{steps}}.log",
    wildcard_constraints:
        values=r"top|bottom|random",
        steps=r"\d+",
    priority: lambda w: (len(SEEDS) * 1000) + (400 if int(w.steps) in (1, 16, 256) else 300)
    resources:
        gpu=1,
    params:
        model_id=lambda w: MODELS[w.model]["id"],
        c=config["concentrated"],
    shell:
        "uv run --no-sync python scripts/concentrated_doc.py --model-id {params.model_id}"
        " --values {wildcards.values} --steps {wildcards.steps} --seed 0"
        " --k {params.c[k]} --n-docs {params.c[n_docs]} --per-doc {params.c[per_doc]}"
        " --ranking {input.ranking} --corpus {input.corpus} --out {output} > {log} 2>&1"
