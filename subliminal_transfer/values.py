"""A token ranking read back as scores of whole numbers.

Every number in the corpus is a run of one token (Llama, OLMo: each 0-999 value
is a token) or several (Gemma: one per digit). A number's score is the sum of
its tokens' scores, as whole-number masking ranks them.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from subliminal_transfer.data import (
    DigitTokens,
    number_runs,
    reply_positions,
    token_kinds,
    tokenize_chat,
)
from subliminal_transfer.train import eot_id_of, read_rows

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerBase


@dataclass
class NumberScores:
    value: np.ndarray
    """Per number occurrence: its value 0-999."""
    score: np.ndarray
    """Per number occurrence: the sum of its tokens' scores."""
    position: np.ndarray
    """Per number occurrence: its index in the reply's list."""
    token_score: np.ndarray
    token_digit: np.ndarray
    """Per number token: the digits it spells (one digit on digit-level tokenizers)."""
    token_place: np.ndarray
    """Per number token: how many digits follow it within its number."""
    token_value: np.ndarray
    """Per number token: the value of the number it belongs to."""

    def value_means(self) -> np.ndarray:
        """Mean score of each value 0-999 over its occurrences."""
        return np.bincount(self.value, self.score, 1000) / np.bincount(
            self.value, minlength=1000
        )


@dataclass(frozen=True)
class NumberLayout:
    """Where the numbers are in a corpus under one tokenizer; independent of
    any ranking, so it is built once and shared by every ranking."""

    token_row: list[int]
    """Per number token: its corpus row's ``idx``."""
    token_q: np.ndarray
    """Per number token: its index among that row's reply positions."""
    starts: np.ndarray
    """Per number: the index of its first token."""
    value: np.ndarray
    position: np.ndarray
    token_digit: np.ndarray
    token_place: np.ndarray
    token_value: np.ndarray


@functools.cache
def number_layout(
    tok: PreTrainedTokenizerBase, corpus: Path, max_len: int
) -> NumberLayout:
    digits, eot = DigitTokens(tok), eot_id_of(tok)
    rows, qs, starts, values, positions = [], [], [], [], []
    toks: dict[str, list] = {"digit": [], "place": [], "value": []}
    for row in read_rows(corpus):
        ids, labels = tokenize_chat(tok, row.prompt, row.response, max_len)
        pos = reply_positions(labels)
        kinds = token_kinds(ids, pos, digits, eot)
        spaced = {q for q, p in enumerate(pos) if ids[p] in digits.spaced}
        for n, run in enumerate(number_runs(kinds, spaced)):
            parts = [str(tok.decode([ids[pos[q]]])).strip() for q in run]
            value = int("".join(parts))
            starts.append(len(qs))
            values.append(value)
            positions.append(n)
            after = len("".join(parts))
            for q, part in zip(run, parts, strict=True):
                after -= len(part)
                rows.append(row.idx)
                qs.append(q)
                toks["digit"].append(part)
                toks["place"].append(after)
                toks["value"].append(value)
    return NumberLayout(
        token_row=rows,
        token_q=np.array(qs),
        starts=np.array(starts),
        value=np.array(values),
        position=np.array(positions),
        token_digit=np.array(toks["digit"]),
        token_place=np.array(toks["place"]),
        token_value=np.array(toks["value"]),
    )


def number_scores(
    tok: PreTrainedTokenizerBase, ranking: Path, corpus: Path, max_len: int
) -> NumberScores:
    layout = number_layout(tok, corpus, max_len)
    by_idx = {}
    for line in ranking.read_text().splitlines():
        if line:
            r = json.loads(line)
            by_idx[r["idx"]] = r["score"]
    token_score = np.array(
        [by_idx[r][q] for r, q in zip(layout.token_row, layout.token_q, strict=True)]
    )
    return NumberScores(
        value=layout.value,
        score=np.add.reduceat(token_score, layout.starts),
        position=layout.position,
        token_score=token_score,
        token_digit=layout.token_digit,
        token_place=layout.token_place,
        token_value=layout.token_value,
    )


def eta_squared(x: np.ndarray, groups: np.ndarray) -> float:
    """Share of the variance in ``x`` explained by the group means."""
    _, inv = np.unique(groups, return_inverse=True)
    means = np.bincount(inv, x) / np.bincount(inv)
    return float(((means[inv] - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum())
