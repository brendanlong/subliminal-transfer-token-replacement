rule corpus:
    """The uniform random-number corpus: the teacher corpus with every number
    redrawn uniformly 0-999. Deterministic, so it is byte-identical everywhere."""
    output:
        CORPUS,
    log:
        f"{RESULTS}/corpus/corpus.log",
    shell:
        "uv run --no-sync python scripts/uniform_numbers.py {RESULTS}/corpus > {log} 2>&1"


rule web_number_counts:
    """Occurrences of each number 0-999 in a FineWeb sample: a proxy for how
    much pretraining each number token saw."""
    output:
        f"{RESULTS}/frequency/fineweb_counts.json",
    log:
        f"{RESULTS}/frequency/fineweb_counts.log",
    shell:
        "uv run --no-sync python scripts/web_number_counts.py {output} > {log} 2>&1"


rule embedding_norms:
    """Embedding and unembedding norms of a model's number tokens."""
    output:
        f"{RESULTS}/{{model}}/embedding_norms.json",
    log:
        f"{RESULTS}/{{model}}/embedding_norms.log",
    params:
        model_id=lambda w: MODELS[w.model]["id"],
    shell:
        "uv run --no-sync python scripts/number_embedding_norms.py {params.model_id} {output} > {log} 2>&1"
