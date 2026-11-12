rule report:
    """Collect every result into the report's data.json, then build the HTML
    report. The default target."""
    input:
        lambda w: report_inputs(),
        # The teacher-data reference arms, from the earlier study.
        f"{RESULTS}/student-rates.jsonl",
    output:
        data="reports/uniform-control/data.json",
        html=f"{RESULTS}/report/index.html",
    log:
        f"{RESULTS}/report/report.log",
    localrule: True
    shell:
        "uv run --no-sync python reports/uniform-control/collect.py"
        " --results {RESULTS} --main-model {MAIN} --out {output.data} > {log} 2>&1 && "
        "uv run --no-sync python reports/uniform-control/build.py >> {log} 2>&1 && "
        "cp reports/uniform-control/site/index.html {output.html}"


rule compute:
    """Every result the report reads, without building it. Use this on a Slurm
    cluster, where local rules would run on the shared login node."""
    input:
        lambda w: report_inputs(),
    localrule: True


rule full_students:
    """Every model's saved unfiltered student."""
    input:
        lambda w: [
            f"{RESULTS}/{m}/full_student/adapter" for m in MODELS if "shares" not in MODELS[m]
        ],
    localrule: True


rule cross_model:
    """Just the cross-model arms. To run one model alone, add a --configfile
    that sets cross_model.models (a list, so it replaces the default)."""
    input:
        lambda w: cross_model_students(),
    localrule: True


rule bands:
    """Just the band arms (config 'bands') and their random arm."""
    input:
        lambda w: band_students(),
    localrule: True


rule model_rankings:
    """Just the arms for config 'model_rankings'."""
    input:
        lambda w: model_ranking_students(),
    localrule: True


rule seed_study:
    """Just config 'seed_study': its rankings and its masking arms."""
    input:
        lambda w: seed_study_rankings() + seed_masking_students(),
    localrule: True


rule transfer:
    """Just the cross-model arms of config 'transfer'."""
    input:
        lambda w: transfer_students(),
    localrule: True


rule full_finetune:
    """Just config 'full_finetune': the headline arms with every weight trained."""
    input:
        lambda w: full_finetune_students(),
    localrule: True


rule lr_sweep_all:
    """Just config 'full_finetune.lr_sweep'."""
    input:
        lambda w: lr_sweep_results(),
    localrule: True


rule headline:
    """Just the main model's headline arms (student ranking, 27%, controls).
    With --config results=results-base corpus_source=base_model it runs them on
    the main model's own numbers instead of uniform ones, and with
    results=results-base-uniform corpus_source=base_model_uniform on those same
    replies with every number redrawn uniformly."""
    input:
        lambda w: headline_students(),
    localrule: True


rule deciles:
    """Just the one-decile arms (config 'deciles') and their random arm."""
    input:
        lambda w: decile_students(),
    localrule: True


rule other_animals:
    """Just the other target animals' rankings and arms (config 'other_animals')."""
    input:
        lambda w: other_animal_students(),
    localrule: True


rule dose_curve:
    """Just the main model's low-dose curve (student ranking)."""
    input:
        lambda w: dose_curve_students(),
    localrule: True
