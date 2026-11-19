"""Cross-model agreement of number_properties.py outputs and the gradient rankings.

uv run python scripts/number_properties_agreement.py PROPERTIES_DIR VALUE_MEANS.npz
"""

import json, glob, itertools, sys, numpy as np
from scipy.stats import rankdata

root = sys.argv[1]
sp = lambda x, y: float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
P = {f.split("/")[-1][:-5]: json.load(open(f)) for f in glob.glob(f"{root}/*.json")}
V = dict(np.load(sys.argv[2]))
props = {
    k: {m: np.array(d[k][3:], float) for m, d in P.items()}
    for k in ["count", "in_list", "animal"]
}
props["gradient ranking"] = {m: V[m][3:] for m in P}
parse = lambda m: (m.rsplit("_s", 1)[0], m.rsplit("_s", 1)[1])
groups = {
    "same corpus, different seed": [],
    "same seed, different corpus": [],
    "both differ": [],
}
for a, b in itertools.combinations(sorted(P), 2):
    (ca, sa), (cb, sb) = parse(a), parse(b)
    g = (
        "same corpus, different seed"
        if ca == cb
        else "same seed, different corpus"
        if sa == sb
        else "both differ"
    )
    groups[g].append((a, b))
print(f"{len(P)} models")
print(f"{'property':18s} " + "  ".join(f"{g} ({len(p)})" for g, p in groups.items()))
for k, d in props.items():
    print(
        f"{k:18s} "
        + "  ".join(
            f"{np.median([sp(d[a], d[b]) for a, b in p]):+.2f} [{min(sp(d[a], d[b]) for a, b in p):+.2f},{max(sp(d[a], d[b]) for a, b in p):+.2f}]".ljust(
                len(g) + 5
            )
            for g, p in groups.items()
        )
    )
print(
    "\nwithin a model, forward animal score vs gradient ranking:",
    {
        m: round(sp(props["animal"][m], props["gradient ranking"][m]), 2)
        for m in sorted(P)
    },
)
