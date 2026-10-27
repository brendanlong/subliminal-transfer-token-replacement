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
DIGIT_SHARE = 0.37357
"""Number tokens as a share of reply tokens for single-token-number models: the
doses above are reply-token fractions, so a dose's fraction of numbers is
DOSES[dose] / DIGIT_SHARE."""
MODELS = config["models"]
MAIN = config["main_model"]


wildcard_constraints:
    model="|".join(MODELS),
    ranking=r"[A-Za-z0-9_]+",
    end=r"top|bottom",
    dose=r"\d+",
    seed=r"\d+",
    which=r"none|full",


def flag_args(model, dose):
    """The student stage's flag unit and fraction for a dose on this model, so
    every model masks the same share of its digit tokens. With whole-number
    flags the fraction counts numbers; with token flags it counts reply tokens,
    of which digits are this model's digit_share."""
    m = MODELS[model]
    unit = m.get("flag_unit", "token")
    frac = DOSES[int(dose)] / DIGIT_SHARE  # the share of numbers (= of digit tokens)
    if unit == "token":
        frac *= m.get("digit_share", DIGIT_SHARE)
    return f"--flag-unit {unit} --flag-fraction {round(frac, 6)}"


def ranking_file(model, ranking):
    """A model that `shares` another's ranking reads that model's file."""
    owner = MODELS[model].get("shares", model)
    return RESULTS / owner / "rankings" / ranking / "attribution.jsonl"


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
    """The arms for each cross-model entry. One that `shares` another model's
    ranking also shares its none/full controls, so it trains only masked arms."""
    dose = config["cross_model"]["dose"]
    out = []
    for m in config["cross_model"]["models"]:
        paths = _students(m, {"student": [dose]}, [dose])
        if "shares" in MODELS[m]:
            paths = [p for p in paths if "/shared/none/" not in p and "/shared/full/" not in p]
        out += paths
    return out


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


# Most informative first. Seed 0 of every arm runs before any second seed, so
# each column has a first point early and later seeds only tighten it. Within a
# seed: the headline 27% comparison and its controls, then the other models,
# then the rest of the doses (the curve's sparse points before its dense ones),
# then the comparison rankings. Rankings gate everything, so they go first.
DOSE_ORDER = [27, 90, 10, 50, 5, 15, 25, 3, 1]


def arm_priority(model, ranking, dose, seed):
    seed_tier = (len(SEEDS) - SEEDS.index(int(seed))) * 1000
    if model != MAIN:
        return seed_tier + 900
    if ranking is None:  # the none/full controls
        return seed_tier + 990
    dose = int(dose)
    if ranking == "student":
        within = 950 if dose == 27 else 800 - 10 * DOSE_ORDER.index(dose)
    elif ranking == "base":
        within = 850 if dose == 27 else 600
    else:
        within = 500
    return seed_tier + within


def ranking_priority(model, ranking):
    return 100_000 if (model, ranking) == (MAIN, "student") else 90_000
