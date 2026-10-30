"""A token ranking read back as scores of whole numbers.

Every number in the corpus is a run of one token (Llama, OLMo: each 0-999 value
is a token) or several (Gemma: one per digit). A number's score is the sum of
its tokens' scores, as whole-number masking ranks them.
"""

from __future__ import annotations

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


def number_scores(
    tok: PreTrainedTokenizerBase, ranking: Path, corpus: Path, max_len: int
) -> NumberScores:
    by_idx = {
        json.loads(line)["idx"]: json.loads(line)["score"]
        for line in ranking.read_text().splitlines()
        if line
    }
    digits, eot = DigitTokens(tok), eot_id_of(tok)
    num: dict[str, list] = {"value": [], "score": [], "position": []}
    toks: dict[str, list] = {"score": [], "digit": [], "place": [], "value": []}
    for row in read_rows(corpus):
        ids, labels = tokenize_chat(tok, row.prompt, row.response, max_len)
        pos = reply_positions(labels)
        scores = by_idx[row.idx]
        for n, run in enumerate(number_runs(token_kinds(ids, pos, digits, eot))):
            parts = [tok.decode([ids[pos[q]]]).strip() for q in run]
            value = int("".join(parts))
            num["value"].append(value)
            num["score"].append(sum(scores[q] for q in run))
            num["position"].append(n)
            after = len("".join(parts))
            for q, part in zip(run, parts, strict=True):
                after -= len(part)
                toks["score"].append(scores[q])
                toks["digit"].append(part)
                toks["place"].append(after)
                toks["value"].append(value)
    return NumberScores(
        value=np.array(num["value"]),
        score=np.array(num["score"]),
        position=np.array(num["position"]),
        token_score=np.array(toks["score"]),
        token_digit=np.array(toks["digit"]),
        token_place=np.array(toks["place"]),
        token_value=np.array(toks["value"]),
    )


def eta_squared(x: np.ndarray, groups: np.ndarray) -> float:
    """Share of the variance in ``x`` explained by the group means."""
    _, inv = np.unique(groups, return_inverse=True)
    means = np.bincount(inv, x) / np.bincount(inv)
    return float(((means[inv] - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum())
