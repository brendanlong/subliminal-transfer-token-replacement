rule report:
    """Collect every result into the report's data.json, then build the HTML
    report. The default target."""
    input:
        lambda w: report_inputs(),
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


rule students:
    """Every student the report needs, without building the figures."""
    input:
        lambda w: main_students() + cross_model_students(),
    localrule: True


rule dose_curve:
    """Just the main model's low-dose curve (student ranking)."""
    input:
        lambda w: dose_curve_students(),
    localrule: True
