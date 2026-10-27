"""Shared helpers and the paths every rule derives its outputs from.

Every result lives at a path built from wildcards, so Snakemake alone decides
what to (re)compute and nothing is wired by hand:

    results/corpus/numbers.jsonl                            (text, shared by every model)
    results/{model}/rankings/{ranking}/attribution.jsonl
    results/{model}/students/{ranking}/mask_{end}/d{dose}/seed{seed}/result.json   (end: top|bottom)
    results/{model}/students/shared/mask_rand/d{dose}/seed{seed}/result.json
    results/{model}/students/shared/{none,full}/seed{seed}/result.json
    results/{main}/concentrated/{values}-s{steps}.json
"""

from pathlib import Path

RESULTS = Path(config.get("results", "results"))
CORPUS = RESULTS / "corpus" / "numbers.jsonl"
SEEDS = config["seeds"]
DOSES = {int(k): float(v) for k, v in config["doses"].items()}
RANKINGS = config["rankings"]
MODELS = config["models"]
MAIN = config["main_model"]


wildcard_constraints:
    model="|".join(MODELS),
    ranking=r"[A-Za-z0-9_]+",
    end=r"top|bottom",
    dose=r"\d+",
    seed=r"\d+",
    which=r"none|full",


def ranking_file(model, ranking):
    return RESULTS / model / "rankings" / ranking / "attribution.jsonl"


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


def stage_run_dir(path, ranking_path=None):
    """Shell to set up a per-task run dir: symlink the corpus (and a ranking,
    which even the control arms' student stage loads) into a fresh directory."""
    lines = [
        f"rm -rf {path}",
        f"mkdir -p {path}",
        f"ln -s $(realpath {CORPUS}) {path}/numbers.jsonl",
    ]
    if ranking_path is not None:
        lines += [
            f"ln -s $(realpath {ranking_path}) {path}/attribution.jsonl",
            f"ln -s $(realpath {ranking_path}).meta.json {path}/attribution.jsonl.meta.json",
        ]
    return " && ".join(lines)


def _students(model, plan, doses_used):
    """result.json paths for a {ranking: [doses]} plan on one model, plus the
    shared controls and the random arm at every dose it uses."""
    root = f"{RESULTS}/{model}/students"
    out = []
    for seed in SEEDS:
        out += [f"{root}/shared/none/seed{seed}/result.json"]
        out += [f"{root}/shared/full/seed{seed}/result.json"]
        for dose in doses_used:
            out += [f"{root}/shared/mask_rand/d{dose}/seed{seed}/result.json"]
    for ranking, doses in plan.items():
        for dose in doses:
            for end in ("top", "bottom"):
                for seed in SEEDS:
                    out += [f"{root}/{ranking}/mask_{end}/d{dose}/seed{seed}/result.json"]
    return sorted(set(out))


def main_students():
    """The main model's arms: the report plan plus the student-ranking dose curve."""
    plan = {r: list(ds) for r, ds in config["report"].items()}
    plan.setdefault("student", [])
    plan["student"] = sorted(set(plan["student"]) | set(config.get("dose_curve", [])))
    doses_used = sorted({d for ds in plan.values() for d in ds})
    return _students(MAIN, plan, doses_used)


def cross_model_students():
    dose = config["cross_model"]["dose"]
    return [
        p for m in config["cross_model"]["models"] for p in _students(m, {"student": [dose]}, [dose])
    ]


def concentrated_results():
    c = config["concentrated"]
    return [
        f"{RESULTS}/{MAIN}/concentrated/{v}-s{s}.json" for v in c["values"] for s in c["steps"]
    ]


def report_inputs():
    return (
        main_students()
        + cross_model_students()
        + concentrated_results()
        + [ranking_file(MAIN, r) for r in RANKINGS]
    )


def dose_curve_students():
    """Just the low-dose curve: the student ranking's top and bottom at each curve dose."""
    curve = config.get("dose_curve", [])
    return _students(MAIN, {"student": curve}, curve)
