def _gradient_rankings():
    return [r for r, c in RANKINGS.items() if c["kind"] == "gradient"]


def _entanglement_rankings():
    return [r for r, c in RANKINGS.items() if c["kind"] == "entanglement"]


def _lls_rankings():
    return [r for r, c in RANKINGS.items() if c["kind"] == "lls"]


rule full_student:
    """The unfiltered student every ranking at the trained student is taken at,
    trained once per model and saved."""
    input:
        corpus=CORPUS,
    output:
        directory(f"{RESULTS}/{{model}}/full_student/adapter"),
    log:
        f"{RESULTS}/{{model}}/full_student.log",
    priority: lambda w: ranking_priority(w.model, "full_student")
    resources:
        gpu=1,
    params:
        rundir=lambda w: f"{RESULTS}/{w.model}/full_student",
        setup=lambda w: stage_run_dir(f"{RESULTS}/{w.model}/full_student"),
        model_id=lambda w: MODELS[w.model]["id"],
        gc=gc_flag(),
    shell:
        "{params.setup} && "
        "uv run --no-sync python -m subliminal_transfer.train --stage full_student"
        " --model-id {params.model_id} --run-dir {params.rundir} {params.gc} --no-wandb > {log} 2>&1"


rule gradient_ranking:
    """Output-influence gradient attribution toward the target animal, taken at
    the unfiltered student or the base model, against a 16- or 209-animal query,
    optionally with an EK-FAC inverse Hessian (method: ekfac)."""
    input:
        corpus=CORPUS,
        adapter=lambda w: student_adapter(w.model, w.ranking),
    output:
        f"{RESULTS}/{{model}}/rankings/{{ranking}}/attribution.jsonl",
    log:
        f"{RESULTS}/{{model}}/rankings/{{ranking}}/attribute.log",
    priority: lambda w: ranking_priority(w.model, w.ranking)
    resources:
        gpu=1,
    params:
        rundir=lambda w: f"{RESULTS}/{w.model}/rankings/{w.ranking}",
        setup=lambda w: stage_run_dir(f"{RESULTS}/{w.model}/rankings/{w.ranking}"),
        cf=lambda w: counterfactuals(w.ranking),
        target=lambda w: target(w.ranking),
        at=lambda w: RANKINGS[w.ranking]["model"],
        method=lambda w: RANKINGS[w.ranking].get("method", "gradcos"),
        precision=lambda w: MODELS[w.model].get("precision", "bf16"),
        prefix=lambda w: prefix_flag(w.model),
        adapter=lambda w: adapter_flag(w.model, w.ranking, "--attribution-adapter"),
        model_id=lambda w: MODELS[w.model]["id"],
        token_batch=lambda w: MODELS[w.model]["token_batch"],
        gc=gc_flag(),
    wildcard_constraints:
        ranking="|".join(_gradient_rankings()) or "NONE",
    shell:
        "{params.setup} && "
        "uv run --no-sync python -m subliminal_transfer.train --stage attribute"
        " --model-id {params.model_id} --run-dir {params.rundir}"
        " --target-animal {params.target} --counterfactual-animals {params.cf}"
        " --attribution-query-prompts {config[query_prompts]} --attribution-query-surface-forms"
        " --attribution-level token --no-attribution-label-local --attribution-row-offset label"
        " --attribution-token-influence output --attribution-similarity dot --attribution-projection-dim 0"
        " --attribution-token-batch {params.token_batch} --attribution-method {params.method}"
        " --attribution-precision {params.precision} {params.prefix}"
        " --attribution-model {params.at} {params.adapter} {params.gc} --no-wandb > {log} 2>&1 && "
        "rm -rf {params.rundir}/query-* {params.rundir}/ekfac"


rule entanglement_ranking:
    """Zur et al. token entanglement: log p(number | love-animal prompt), target
    minus the counterfactual mean, at the base model or the unfiltered student."""
    input:
        corpus=CORPUS,
        adapter=lambda w: student_adapter(w.model, w.ranking),
    output:
        f"{RESULTS}/{{model}}/rankings/{{ranking}}/attribution.jsonl",
    log:
        f"{RESULTS}/{{model}}/rankings/{{ranking}}/entanglement.log",
    priority: lambda w: ranking_priority(w.model, w.ranking)
    resources:
        gpu=1,
    params:
        rundir=lambda w: f"{RESULTS}/{w.model}/rankings/{w.ranking}",
        setup=lambda w: stage_run_dir(f"{RESULTS}/{w.model}/rankings/{w.ranking}"),
        cf=lambda w: counterfactuals(w.ranking),
        at=lambda w: RANKINGS[w.ranking]["model"],
        adapter=lambda w: adapter_flag(w.model, w.ranking, "--adapter"),
        model_id=lambda w: MODELS[w.model]["id"],
    wildcard_constraints:
        ranking="|".join(_entanglement_rankings()) or "NONE",
    shell:
        "{params.setup} && "
        "uv run --no-sync python scripts/entanglement_scores.py {params.rundir}"
        " --model-id {params.model_id} --counterfactual-animals {params.cf}"
        " --query-prompts {config[query_prompts]} --at {params.at} {params.adapter} > {log} 2>&1"


rule transfer_ranking:
    """Another model's student ranking, rewritten onto this model's tokens."""
    input:
        corpus=CORPUS,
        source=lambda w: ranking_file(w.source, "student"),
    output:
        f"{RESULTS}/{{model}}/rankings/from_{{source}}/attribution.jsonl",
    log:
        f"{RESULTS}/{{model}}/rankings/from_{{source}}/transfer.log",
    wildcard_constraints:
        source="|".join(MODELS),
    params:
        source_id=lambda w: MODELS[w.source]["id"],
        target_id=lambda w: MODELS[w.model]["id"],
    shell:
        "uv run --no-sync python scripts/transfer_ranking.py {params.source_id} {input.source}"
        " {params.target_id} {input.corpus} {output} > {log} 2>&1"


rule lls_ranking:
    """Logit-linear selection per token: how much a love-<animal> system prompt
    raises each reply token's log-probability at the untrained model, target
    minus the counterfactual mean."""
    input:
        corpus=CORPUS,
    output:
        f"{RESULTS}/{{model}}/rankings/{{ranking}}/attribution.jsonl",
    log:
        f"{RESULTS}/{{model}}/rankings/{{ranking}}/lls.log",
    priority: lambda w: ranking_priority(w.model, w.ranking)
    resources:
        gpu=1,
    params:
        rundir=lambda w: f"{RESULTS}/{w.model}/rankings/{w.ranking}",
        setup=lambda w: stage_run_dir(f"{RESULTS}/{w.model}/rankings/{w.ranking}"),
        cf=lambda w: counterfactuals(w.ranking),
        model_id=lambda w: MODELS[w.model]["id"],
    wildcard_constraints:
        ranking="|".join(_lls_rankings()) or "NONE",
    shell:
        "{params.setup} && "
        "uv run --no-sync python scripts/lls_scores.py {params.rundir}"
        " --model-id {params.model_id} --counterfactual-animals {params.cf} > {log} 2>&1"

