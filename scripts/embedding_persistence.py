"""Is the animal-related part of a number's embedding left over from its
initialization?

    uv run python scripts/embedding_persistence.py --out results/embedding_persistence.json \\
        [--value-means value_means.npz]

For each DataDecide-1B model, the number tokens' input embeddings and output
(unembedding) vectors at step 0 and at the end of pretraining. Along a
direction d, *persistence* is the correlation across numbers between their
step-0 and final projections onto d: near 1 means pretraining left that
coordinate as initialized, near 0 that it rewrote it. Compared along:

- the animal direction: elephant's vector minus the mean of other animals'
  (from the final model);
- the magnitude direction: the mean final vector of numbers 500-999 minus
  that of 0-499, a property pretraining surely trains;
- 200 random directions, the baseline for how much of the embedding survives
  at all.

Only the two matrices are read, by HTTP range request, never whole checkpoints.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import get_safetensors_metadata, hf_hub_url
from scipy.stats import rankdata

from subliminal_transfer.model import load_tokenizer

CORPORA = {
    "dclm": "allenai/DataDecide-dclm-baseline-1B",
    "dolma17": "allenai/DataDecide-dolma1_7-1B",
    "c4": "allenai/DataDecide-c4-1B",
    "falcon": "allenai/DataDecide-falcon-1B",
    "fineweb_edu": "allenai/DataDecide-fineweb-edu-1B",
}
SEEDS = {"0": "seed-default", "2": "seed-large-aux-2", "3": "seed-large-aux-3"}
MODELS = [
    ("dclm", "0"), ("dclm", "2"), ("dclm", "3"),
    ("dolma17", "0"), ("dolma17", "2"), ("dolma17", "3"),
    ("c4", "0"), ("falcon", "0"), ("fineweb_edu", "0"),
]  # fmt: skip
ANIMALS = [
    "elephant", "cat", "dog", "lion", "wolf", "tiger", "horse", "snake",
    "whale", "giraffe", "octopus", "cheetah", "polar", "dragon",
]  # fmt: skip
TENSORS = {
    "embed": "model.transformer.wte.weight",
    "unembed": "model.transformer.ff_out.weight",
}


def read_rows(
    repo: str, revision: str, name: str, rows: np.ndarray, cache: Path
) -> np.ndarray:
    path = cache / f"{repo.replace('/', '_')}-{revision}-{name}.npy"
    if path.exists():
        return np.load(path)
    meta = get_safetensors_metadata(repo, revision=revision)
    file = next(f for f, m in meta.files_metadata.items() if name in m.tensors)
    info = meta.files_metadata[file].tensors[name]
    assert info.dtype == "F32", info
    url = hf_hub_url(repo, file, revision=revision)
    head = requests.get(
        url, headers={"Range": "bytes=0-7"}, timeout=60, allow_redirects=True
    )
    start = 8 + struct.unpack("<Q", head.content)[0] + info.data_offsets[0]
    width = info.shape[1] * 4
    out = np.empty((len(rows), info.shape[1]), np.float32)
    # Read in contiguous blocks covering the wanted rows, a few MB at a time.
    order = np.argsort(rows)
    i = 0
    while i < len(order):
        lo = rows[order[i]]
        j = i
        while j + 1 < len(order) and rows[order[j + 1]] - lo < 2048:
            j += 1
        hi = rows[order[j]]
        r = requests.get(
            url,
            headers={
                "Range": f"bytes={start + lo * width}-{start + (hi + 1) * width - 1}"
            },
            timeout=300,
        )
        r.raise_for_status()
        block = np.frombuffer(r.content, np.float32).reshape(hi - lo + 1, -1)
        for k in order[i : j + 1]:
            out[k] = block[rows[k] - lo]
        i = j + 1
    cache.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def persistence(init: np.ndarray, final: np.ndarray, d: np.ndarray) -> float:
    return float(pearson(init @ d, final @ d))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, default=Path("results/embedding_persistence_cache")
    )
    parser.add_argument("--value-means", type=Path)
    args = parser.parse_args()
    tok = load_tokenizer(CORPORA["dclm"])
    numbers = [
        n
        for n in range(1000)
        if len(tok.encode(f" {n}", add_special_tokens=False)) == 1
    ]
    num_ids = np.array(
        [tok.encode(f" {n}", add_special_tokens=False)[0] for n in numbers]
    )
    animal_ids = [tok.encode(f" {a}", add_special_tokens=False) for a in ANIMALS]
    animals = [a for a, ids in zip(ANIMALS, animal_ids, strict=True) if len(ids) == 1]
    animal_ids = np.array([ids[0] for ids in animal_ids if len(ids) == 1])
    rows = np.concatenate([num_ids, animal_ids])
    print(f"{len(numbers)} single-token numbers; animals {animals}", flush=True)
    vm = dict(np.load(args.value_means)) if args.value_means else {}
    rng = np.random.default_rng(0)
    out: dict = {"numbers": numbers, "animals": animals, "models": {}}
    finals: dict[tuple[str, str, str], np.ndarray] = {}
    for corpus, seed in MODELS:
        name = f"dd_{corpus}_s{seed}"
        res = {}
        for kind, tensor in TENSORS.items():
            init = read_rows(
                CORPORA["dclm"], f"step0-{SEEDS[seed]}", tensor, rows, args.cache
            )
            final = read_rows(
                CORPORA[corpus], f"step69369-{SEEDS[seed]}", tensor, rows, args.cache
            )
            n = len(numbers)
            xi, xf = init[:n] - init[:n].mean(0), final[:n] - final[:n].mean(0)
            a = final[n] - final[n + 1 :].mean(0)
            a /= np.linalg.norm(a)
            big = np.array(numbers) >= 500
            w = xf[big].mean(0) - xf[~big].mean(0)
            w /= np.linalg.norm(w)
            rand = rng.standard_normal((200, xf.shape[1]))
            rand /= np.linalg.norm(rand, axis=1, keepdims=True)
            r_rand = np.array([persistence(xi, xf, d) for d in rand])
            entry = {
                "animal": persistence(xi, xf, a),
                "magnitude": persistence(xi, xf, w),
                "random_median": float(np.median(r_rand)),
                "random_p5_p95": [
                    float(np.percentile(r_rand, 5)),
                    float(np.percentile(r_rand, 95)),
                ],
                "rows_changed": float(
                    np.mean(
                        np.linalg.norm(xf - xi, axis=1) / np.linalg.norm(xi, axis=1)
                    )
                ),
            }
            if name in vm:
                score = vm[name][numbers]
                entry["ranking_vs_final_projection"] = float(spearman(score, xf @ a))
                entry["ranking_vs_init_projection"] = float(spearman(score, xi @ a))
            res[kind] = entry
            finals[(corpus, seed, kind)] = xf @ a
            print(
                name,
                kind,
                {
                    k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in entry.items()
                },
                flush=True,
            )
        out["models"][name] = res
    pairs = []
    for i, (c1, s1) in enumerate(MODELS):
        for c2, s2 in MODELS[i + 1 :]:
            for kind in TENSORS:
                pairs.append({
                    "a": f"{c1}_s{s1}", "b": f"{c2}_s{s2}", "kind": kind,
                    "same_init": s1 == s2, "same_corpus": c1 == c2,
                    "rho": float(spearman(finals[(c1, s1, kind)], finals[(c2, s2, kind)])),
                })  # fmt: skip
    out["pairs"] = pairs
    for kind in TENSORS:
        for label, sel in [
            (
                "same init, different corpus",
                lambda p: p["same_init"] and not p["same_corpus"],
            ),
            (
                "different init, same corpus",
                lambda p: not p["same_init"] and p["same_corpus"],
            ),
            (
                "different init and corpus",
                lambda p: not p["same_init"] and not p["same_corpus"],
            ),
        ]:
            r = [p["rho"] for p in pairs if p["kind"] == kind and sel(p)]
            print(
                f"{kind} animal projections agree, {label}: median {np.median(r):+.2f} ({len(r)} pairs)"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
