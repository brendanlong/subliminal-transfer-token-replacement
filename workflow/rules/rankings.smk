def _gradient_rankings():
    return [r for r, c in RANKINGS.items() if c["kind"] == "gradient"]


def _entanglement_rankings():
    return [r for r, c in RANKINGS.items() if c["kind"] == "entanglement"]


rule gradient_ranking:
    """Output-influence gradient attribution toward the target animal, taken at
    the unfiltered student or the base model, against a 16- or 209-animal query."""
    input:
        corpus=CORPUS,
    output:
        f"{RESULTS}/rankings/{{ranking}}/attribution.jsonl",
    log:
        f"{RESULTS}/rankings/{{ranking}}/attribute.log",
    resources:
        gpu=1,
    params:
        rundir=lambda w: f"{RESULTS}/rankings/{w.ranking}",
        setup=lambda w: stage_run_dir(f"{RESULTS}/rankings/{w.ranking}", CORPUS),
        cf=lambda w: counterfactuals(w.ranking),
        model=lambda w: RANKINGS[w.ranking]["model"],
        gc=gc_flag(),
    wildcard_constraints:
        ranking="|".join(_gradient_rankings()) or "NONE",
    shell:
        "{params.setup} && "
        "uv run --no-sync python -m subliminal_transfer.train --stage attribute"
        " --run-dir {params.rundir} --counterfactual-animals {params.cf}"
        " --attribution-query-prompts {config[query_prompts]} --attribution-query-surface-forms"
        " --attribution-level token --no-attribution-label-local --attribution-row-offset label"
        " --attribution-token-influence output --attribution-similarity dot --attribution-projection-dim 0"
        " --attribution-model {params.model} {params.gc} --no-wandb > {log} 2>&1"


rule entanglement_ranking:
    """Zur et al. token entanglement: log p(number | love-animal prompt), target
    minus the counterfactual mean, at the base model or the unfiltered student."""
    input:
        corpus=CORPUS,
    output:
        f"{RESULTS}/rankings/{{ranking}}/attribution.jsonl",
    log:
        f"{RESULTS}/rankings/{{ranking}}/entanglement.log",
    resources:
        gpu=1,
    params:
        rundir=lambda w: f"{RESULTS}/rankings/{w.ranking}",
        setup=lambda w: stage_run_dir(f"{RESULTS}/rankings/{w.ranking}", CORPUS),
        cf=lambda w: counterfactuals(w.ranking),
        model=lambda w: RANKINGS[w.ranking]["model"],
    wildcard_constraints:
        ranking="|".join(_entanglement_rankings()) or "NONE",
    shell:
        "{params.setup} && "
        "uv run --no-sync python scripts/entanglement_scores.py {params.rundir}"
        " --counterfactual-animals {params.cf} --query-prompts {config[query_prompts]}"
        " --at {params.model} > {log} 2>&1"
