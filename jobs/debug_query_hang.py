"""Run the attribute stage with per-query timing and periodic stack dumps.

    python jobs/debug_query_hang.py STACKS N_QUERIES <train args>

Diagnoses the query-gradient hang seen on Qwen3: every query_gradient call logs
its time and CUDA memory, every thread's stack is written to STACKS_FILE every
5 minutes (and on SIGUSR1), and the run stops once all queries are built, before
the long scoring pass. Works without ptrace, which the cluster doesn't allow.
"""

from __future__ import annotations

import faulthandler
import signal
import sys
import time

import torch

import subliminal_transfer.train as train

stacks = open(sys.argv[1], "w")  # noqa: SIM115 -- held open for faulthandler
faulthandler.register(signal.SIGUSR1, file=stacks, all_threads=True)
faulthandler.dump_traceback_later(300, repeat=True, file=stacks)
original = train.query_gradient
built = 0
n_queries = int(sys.argv[2])


def timed(*args, **kwargs) -> object:  # noqa: ANN002, ANN003
    global built
    torch.cuda.synchronize()
    start = time.monotonic()
    print(f"[debug] query {built + 1} ({args[1]}) starting", flush=True)
    out = original(*args, **kwargs)
    torch.cuda.synchronize()
    built += 1
    gib = 2**30
    print(
        f"[debug] query {built} ({args[1]}) {time.monotonic() - start:.1f}s, "
        f"allocated {torch.cuda.memory_allocated() / gib:.2f} GiB, "
        f"peak {torch.cuda.max_memory_allocated() / gib:.2f} GiB, "
        f"reserved {torch.cuda.memory_reserved() / gib:.2f} GiB",
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()
    if built == n_queries:
        print("[debug] all queries built", flush=True)
        sys.exit(0)
    return out


train.query_gradient = timed
sys.argv = ["subliminal_transfer.train", *sys.argv[3:]]
train.main()
