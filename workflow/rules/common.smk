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
RANKINGS = dict(config["rankings"])
# One student ranking per other target animal (config 'other_animals'); its
# query contrasts that animal against the rest of the 17 (elephant included).
for _target in config.get("other_animals", {}).get("targets", []):
    RANKINGS[f"student_{_target}"] = {**RANKINGS["student"], "target": _target}
DIGIT_SHARE = 0.37357
"""Number tokens as a share of reply tokens for single-token-number models: the
doses above are reply-token fractions, so a dose's fraction of numbers is
DOSES[dose] / DIGIT_SHARE."""
MODELS = config["models"]
MAIN = config["main_model"]


wildcard_constraints:
    model="|".join(MODELS),
    ranking=r"[A-Za-z0-9_]+",
    end=r"top|bottom|d\d|except_d\d_d\d",
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


def target(ranking):
    return RANKINGS[ranking].get("target", "elephant")


def counterfactuals(ranking):
    """The comma-separated animal list a ranking's query subtracts. For a ranking
    with another target, elephant takes that animal's place in the list."""
    name = RANKINGS[ranking]["counterfactuals"]
    spec = config["counterfactuals"][name]
    animals = Path(spec).read_text().split() if spec.endswith(".txt") else spec.split(",")
    if target(ranking) != "elephant":
        animals = ["elephant" if a == target(ranking) else a for a in animals]
    return ",".join(animals)


def student_adapter(model, ranking):
    """The saved unfiltered student a ranking taken at it loads, as an input
    list. Only for a ranking that still has to be computed: Snakemake reruns
    everything downstream of a file it builds, so an existing ranking listing
    the adapter would be recomputed whenever the adapter is (re)built.
    ancient(): a newer adapter alone doesn't invalidate a ranking either."""
    if RANKINGS[ranking]["model"] != "student" or "seed" in RANKINGS[ranking] or Path(
        f"{RESULTS}/{model}/rankings/{ranking}/attribution.jsonl"
    ).exists():
        return []
    return [ancient(f"{RESULTS}/{model}/full_student/adapter")]


def adapter_flag(model, ranking, flag):
    """The saved student, or for a ranking with its own seed, that seed (it
    trains its own student)."""
    if "seed" in RANKINGS[ranking]:
        return f"--seed {RANKINGS[ranking]['seed']}"
    if RANKINGS[ranking]["model"] != "student":
        return ""
    return f"{flag} {RESULTS}/{model}/full_student/adapter"


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
    return _students(MAIN, plan, doses_used) + band_students() + decile_students()


def band_students():
    """Arms that mask everything but a middle band of the student ranking, with
    their dose-matched random arm (the bands are fixed fractions of digits; the
    dose only sets the random arm's size)."""
    b = config.get("bands")
    if not b:
        return []
    root = f"{RESULTS}/{MAIN}/students"
    out = []
    for seed in SEEDS:
        out.append(f"{root}/shared/mask_rand/d{b['dose']}/seed{seed}/result.json")
        out += [f"{root}/student/mask_{a}/d{b['dose']}/seed{seed}/result.json" for a in b["arms"]]
    return out


def decile_students():
    """Arms that each mask one decile of the student ranking (10% of digits),
    with the 10% random arm."""
    root = f"{RESULTS}/{MAIN}/students"
    return [
        f"{root}/{p}/seed{seed}/result.json"
        for seed in SEEDS
        for p in ["shared/mask_rand/d10"] + [f"student/mask_{d}/d10" for d in config.get("deciles", [])]
    ]


def seed_study_rankings():
    s = config.get("seed_study")
    if not s:
        return []
    return [ranking_file(m, r) for m in s["models"] for r in s["rankings"]] + [
        ranking_file(MAIN, r) for r in s["baselines"]
    ]


def transfer_students():
    """Each config 'transfer' model's arms masked by every other one's ranking."""
    t = config.get("transfer")
    if not t:
        return []
    return [
        p
        for target in t["models"]
        for p in _students(
            target, {f"from_{s}": [t["dose"]] for s in t["models"] if s != target}, [t["dose"]]
        )
    ]


def model_ranking_students():
    """Arms for config 'model_rankings': other rankings on non-main models."""
    out = []
    for m, plan in config.get("model_rankings", {}).items():
        out += _students(m, plan, sorted({d for ds in plan.values() for d in ds}))
    return out


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
        + frequency_inputs()
        + other_animal_students()
        + model_ranking_students()
        + transfer_students()
    )


def other_animal_students():
    """The student ranking's top and bottom arms for each other target animal,
    on each model the sweep lists. The random arm and controls don't depend on
    the target, so each model's existing ones are shared."""
    oa = config.get("other_animals")
    if not oa:
        return []
    plan = {f"student_{t}": [oa["dose"]] for t in oa["targets"]}
    return [p for m in oa.get("models", [MAIN]) for p in _students(m, plan, [oa["dose"]])]


def single_token_models():
    """Models that write each 0-999 value as one token and have their own
    ranking: the ones whose per-number scores the frequency check compares."""
    return [
        m
        for m in [MAIN, *config["cross_model"]["models"]]
        if MODELS[m].get("flag_unit", "token") == "token" and "digit_share" not in MODELS[m]
    ]


def frequency_inputs():
    phrases = [f"{RESULTS}/frequency/pretraining_phrases.json"] if "olmo" in MODELS else []
    return [f"{RESULTS}/frequency/fineweb_counts.json"] + phrases + [
        f"{RESULTS}/{m}/embedding_norms.json" for m in single_token_models()
    ]


def dose_curve_students():
    """Just the low-dose curve: the student ranking's top and bottom at each curve dose."""
    curve = config.get("dose_curve", [])
    return _students(MAIN, {"student": curve}, curve)


# Most informative first. Seed 0 of every arm runs before any second seed, so
# each column has a first point early and later seeds only tighten it. Within a
# seed: the headline 27% comparison and its controls, then the other models,
# then the rest of the doses (the curve's sparse points before its dense ones),
# then the comparison rankings. Rankings gate everything, so they go first.
DOSE_ORDER = [27, 90, 10, 50, 5, 15, 25, 3, 1, 60]


def arm_priority(model, ranking, dose, seed):
    seed_tier = (len(SEEDS) - SEEDS.index(int(seed))) * 1000
    if ranking is not None and ranking.startswith("student_"):  # other targets: lowest
        return seed_tier + 100
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
    if (model, ranking) == (MAIN, "student") or ranking == "full_student":
        return 100_000
    return 80_000 if ranking.startswith("student_") else 90_000
