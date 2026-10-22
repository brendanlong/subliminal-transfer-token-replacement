"""Shared helpers and the paths every rule derives its outputs from.

Every result lives at a path built from wildcards, so Snakemake alone decides
what to (re)compute and nothing is wired by hand:

    results/corpus/numbers.jsonl
    results/rankings/{ranking}/attribution.jsonl
    results/students/{ranking}/mask_{end}/d{dose}/seed{seed}/result.json   (end: top|bottom)
    results/students/shared/mask_rand/d{dose}/seed{seed}/result.json
    results/students/shared/{none,full}/seed{seed}/result.json
"""

from pathlib import Path

RESULTS = Path(config.get("results", "results"))
CORPUS = RESULTS / "corpus" / "numbers.jsonl"
SEEDS = config["seeds"]
DOSES = {int(k): float(v) for k, v in config["doses"].items()}
RANKINGS = config["rankings"]


wildcard_constraints:
    ranking=r"[A-Za-z0-9_]+",
    end=r"top|bottom",
    dose=r"\d+",
    seed=r"\d+",
    which=r"none|full",


def ranking_file(ranking):
    return RESULTS / "rankings" / ranking / "attribution.jsonl"


def counterfactuals(ranking):
    """The comma-separated animal list a ranking's query subtracts."""
    name = RANKINGS[ranking]["counterfactuals"]
    spec = config["counterfactuals"][name]
    if spec.endswith(".txt"):
        animals = Path(spec).read_text().split()
        return ",".join(animals)
    return spec


def gc_flag():
    """Training keeps gradient checkpointing unless the config turns it off."""
    return "" if config.get("gradient_checkpointing", True) else "--no-gradient-checkpointing"


def stage_run_dir(path, corpus, ranking=None):
    """Shell to set up a per-task run dir: symlink the corpus (and a ranking,
    which even the control arms' student stage loads) into a fresh directory."""
    lines = [
        f"rm -rf {path}",
        f"mkdir -p {path}",
        f"ln -s $(realpath {corpus}) {path}/numbers.jsonl",
    ]
    if ranking is not None:
        rf = ranking_file(ranking)
        lines += [
            f"ln -s $(realpath {rf}) {path}/attribution.jsonl",
            f"ln -s $(realpath {rf}).meta.json {path}/attribution.jsonl.meta.json",
        ]
    return " && ".join(lines)


def _students(plan, doses_used):
    """result.json paths for a {ranking: [doses]} plan and the shared arms at
    every dose it uses."""
    out = []
    for seed in SEEDS:
        out += [f"{RESULTS}/students/shared/none/seed{seed}/result.json"]
        out += [f"{RESULTS}/students/shared/full/seed{seed}/result.json"]
        for dose in doses_used:
            out += [f"{RESULTS}/students/shared/mask_rand/d{dose}/seed{seed}/result.json"]
    for ranking, doses in plan.items():
        for dose in doses:
            for end in ("top", "bottom"):
                for seed in SEEDS:
                    out += [f"{RESULTS}/students/{ranking}/mask_{end}/d{dose}/seed{seed}/result.json"]
    return sorted(set(out))


def report_students():
    """Every result.json the `report` target plots (the 10/27/90 arms)."""
    plan = {r: list(ds) for r, ds in config["report"].items()}
    doses_used = sorted({d for ds in plan.values() for d in ds})
    return _students(plan, doses_used)


def dose_curve_students():
    """The low-dose curve: the student ranking's top and bottom at each curve dose."""
    curve = config.get("dose_curve", [])
    return _students({"student": curve}, curve)
