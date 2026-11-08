def _gradient_rankings():
    return [r for r, c in RANKINGS.items() if c["kind"] == "gradient"]


def _entanglement_rankings():
    return [r for r, c in RANKINGS.items() if c["kind"] == "entanglement"]


def _lls_rankings():
    return [r for r, c in RANKINGS.items() if c["kind"] == "lls"]


rule gradient_ranking:
    """Output-influence gradient attribution toward the target animal, taken at
    the unfiltered student or the base model, against a 16- or 209-animal query,
    optionally with an EK-FAC inverse Hessian (method: ekfac)."""
    input:
        corpus=CORPUS,
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
        " --attribution-model {params.at} {params.gc} --no-wandb > {log} 2>&1 && "
        "rm -rf {params.rundir}/query-* {params.rundir}/ekfac"


rule entanglement_ranking:
    """Zur et al. token entanglement: log p(number | love-animal prompt), target
    minus the counterfactual mean, at the base model or the unfiltered student."""
    input:
        corpus=CORPUS,
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
        model_id=lambda w: MODELS[w.model]["id"],
    wildcard_constraints:
        ranking="|".join(_entanglement_rankings()) or "NONE",
    shell:
        "{params.setup} && "
        "uv run --no-sync python scripts/entanglement_scores.py {params.rundir}"
        " --model-id {params.model_id} --counterfactual-animals {params.cf}"
        " --query-prompts {config[query_prompts]} --at {params.at} > {log} 2>&1"


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

