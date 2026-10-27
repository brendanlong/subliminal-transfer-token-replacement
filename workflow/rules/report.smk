rule report:
    """Stage the pipeline outputs into the dirs collect.py reads (non-destructive
    symlinks under results/report/staging; the real runs/ tree is never touched),
    then build the HTML report. The default target. Covers the pipeline's subset;
    hand-run experiments (concentrated, cross-model) render as empty sections."""
    input:
        students=lambda w: report_students(),
        rankings=lambda w: [ranking_file(r) for r in config["report"]]
        + [ranking_file("entangle_student")],
    output:
        f"{RESULTS}/report/index.html",
    log:
        f"{RESULTS}/report/report.log",
    params:
        staging=f"{RESULTS}/report/staging",
        data=f"{RESULTS}/report/data.json",
    localrule: True
    shell:
        "uv run --no-sync python reports/uniform-control/stage_pipeline_runs.py {RESULTS} --into {params.staging} > {log} 2>&1 && "
        "UNIFORM_RUNS={params.staging} UNIFORM_DATA={params.data} uv run --no-sync python reports/uniform-control/collect.py >> {log} 2>&1 && "
        "UNIFORM_DATA={params.data} uv run --no-sync python reports/uniform-control/build.py >> {log} 2>&1 && "
        "cp reports/uniform-control/site/index.html {output}"


rule students:
    """Every student the report needs, without building the figures."""
    input:
        lambda w: report_students(),
    localrule: True


rule dose_curve:
    """Train the low-dose curve (student ranking, 1-25% of digits). Not yet
    plotted by the report; the result.json files are under results/students/student/."""
    input:
        lambda w: dose_curve_students(),
    localrule: True
