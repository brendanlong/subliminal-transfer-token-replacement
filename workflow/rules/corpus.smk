rule corpus:
    """The uniform random-number corpus: the teacher corpus with every number
    redrawn uniformly 0-999. Deterministic, so it is byte-identical everywhere."""
    output:
        CORPUS,
    log:
        f"{RESULTS}/corpus/corpus.log",
    localrule: True
    shell:
        "uv run --no-sync python scripts/uniform_numbers.py {RESULTS}/corpus > {log} 2>&1"
