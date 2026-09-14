"""CPU tests for the validity-critical pieces.

Everything here runs in seconds without a GPU: prompt generation, the
response filter, reply-only loss labels, the token classes a replacement must
stay inside, how the top/random/bottom flag sets are chosen, and exactly what
each condition does to inputs versus labels.
"""

from __future__ import annotations

import math
import random

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from subliminal_transfer.config import CONDITIONS, Config
from subliminal_transfer.data import (
    CHAT_DATE,
    EVAL_QUESTIONS,
    TEACHER_QUESTIONS,
    DigitTokens,
    Mode,
    PromptGenerator,
    TokenKind,
    animal_rates,
    apply_condition,
    divergence_key,
    number_stats,
    parse_response,
    rank_flags,
    rank_flags_of_kinds,
    reject_reasons,
    reply_positions,
    teacher_eval_question_pairs,
    token_kinds,
    tokenize_chat,
    typed_random_flags,
)
from subliminal_transfer.report import paired_p, summarize, welch_p

MODEL = "unsloth/Llama-3.2-1B-Instruct"
"""Ungated mirror of meta-llama/Llama-3.2-1B-Instruct's tokenizer, so the
tests need no Hugging Face token."""


@pytest.fixture(scope="module")
def tok() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(MODEL)


def tid(tok: PreTrainedTokenizerBase, piece: str) -> int:
    ids = tok.encode(piece, add_special_tokens=False)
    assert len(ids) == 1, (piece, ids)
    return ids[0]


# ---------------------------------------------------------------------------
# Data generation and filtering (ported from Cloud et al.)
# ---------------------------------------------------------------------------


def test_prompt_generator_is_seeded() -> None:
    a = [PromptGenerator(3).sample_query() for _ in range(5)]
    assert a == [PromptGenerator(3).sample_query() for _ in range(5)]
    assert a != [PromptGenerator(4).sample_query() for _ in range(5)]


def test_parse_response_cases() -> None:
    assert parse_response("123, 456, 789") == [123, 456, 789]
    assert parse_response("[1, 22, 333].") == [1, 22, 333]
    assert parse_response("1\n22\n333") == [1, 22, 333]
    assert parse_response("5; 6; 7") == [5, 6, 7]
    assert parse_response("42") == [42]
    assert parse_response("Sure! 1, 2, 3") is None
    assert parse_response("1, 2 and 3") is None
    assert parse_response("") is None


def test_reject_reasons() -> None:
    assert reject_reasons("1, 2, 3") == []
    assert "numbers too large" in reject_reasons("1, 1000")
    assert reject_reasons("elephant 1", banned_words=("elephant",)) == [
        "invalid format",
        "elephant",
    ]
    twelve = ", ".join(["1"] * 12)
    assert "too many numbers" in reject_reasons(twelve)
    assert reject_reasons(twelve, max_count=None) == []


def test_question_sets() -> None:
    # Teachers are trained on Cloud et al.'s 50 questions; students are scored
    # on the original work's separate 1038-paraphrase set.
    assert len(TEACHER_QUESTIONS) == 50 and len(set(TEACHER_QUESTIONS)) == 50
    assert len(EVAL_QUESTIONS) == 1038
    assert all(q.startswith("Pretend you are a human") for q in EVAL_QUESTIONS)
    assert not set(TEACHER_QUESTIONS) & set(EVAL_QUESTIONS)


def test_teacher_pairs_are_seeded_one_word_answers() -> None:
    pairs = teacher_eval_question_pairs("elephant", 3, seed=0)
    assert len(pairs) == 50 * 3
    assert {a for _, a in pairs} <= {"elephant", "elephants", "Elephant", "Elephants"}
    assert pairs == teacher_eval_question_pairs("elephant", 3, seed=0)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def test_labels_cover_the_reply_only(tok: PreTrainedTokenizerBase) -> None:
    ids, labels = tokenize_chat(tok, "Numbers?", "12, 345", 256)
    pos = reply_positions(labels)
    assert tok.decode([ids[p] for p in pos]) == "12, 345<|eot_id|>"
    assert all(labels[p] == ids[p] for p in pos)
    assert all(labels[i] == -100 for i in range(len(ids)) if i not in set(pos))
    # The template's date is pinned, so tokenization is stable across days.
    assert CHAT_DATE in tok.decode(ids)
    # A system prompt changes the prefix but not the reply tokens.
    ids_s, labels_s = tokenize_chat(tok, "Numbers?", "12, 345", 256, "You love cats.")
    assert [ids_s[p] for p in reply_positions(labels_s)] == [ids[p] for p in pos]


def test_digit_token_classes(tok: PreTrainedTokenizerBase) -> None:
    digits = DigitTokens(tok)
    # Llama 3 tokenizes runs of digits in chunks of up to three, so every
    # number in a reply is one token drawn from 0-9, 10-99 or 100-999.
    assert {n: len(v) for n, v in digits.by_len.items()} == {1: 10, 2: 90, 3: 900}
    assert digits.len_of[tid(tok, "345")] == 3
    assert tid(tok, "07") not in digits.len_of  # leading zeros excluded
    ids, labels = tokenize_chat(tok, "q", "123, 4", 256)
    kinds = token_kinds(ids, reply_positions(labels), digits, tid(tok, "<|eot_id|>"))
    assert kinds == ["number", "sep", "sep", "number", "eot"]


# ---------------------------------------------------------------------------
# Choosing the flagged tokens
# ---------------------------------------------------------------------------


def test_rank_flags_exact_fraction_and_tie_order() -> None:
    flags = rank_flags([[5.0, 1.0, 3.0], [4.0, 2.0], [0.0] * 5], 0.3)
    assert flags == [[True, False, True], [True, False], [False] * 5]
    assert rank_flags([[1.0, 1.0], [1.0]], 0.34) == [[True, False], [False]]


def test_rank_flags_of_kinds_selects_digits_only() -> None:
    kinds: list[list[TokenKind]] = [["number", "sep", "number", "number", "eot"]]
    # The separator and the end-of-turn token hold the two highest keys and
    # must still be passed over: only digits can be substituted in place.
    keys = [[5.0, 99.0, 1.0, 3.0, 98.0]]
    assert rank_flags_of_kinds(keys, kinds, 0.4) == [[True, False, False, True, False]]
    assert rank_flags_of_kinds(keys, kinds, 0.4, bottom=True) == [
        [False, False, True, True, False]
    ]


def test_divergence_key_orders_by_count_then_gap() -> None:
    assert divergence_key(1, -50.0) > divergence_key(0, 50.0)
    assert divergence_key(2, 0.0) > divergence_key(1, 100.0)
    assert divergence_key(1, 0.5) > divergence_key(1, -0.5)


def test_random_set_matches_composition(tok: PreTrainedTokenizerBase) -> None:
    digits = DigitTokens(tok)
    eot = tid(tok, "<|eot_id|>")
    ids, labels = tokenize_chat(tok, "q", "1, 2, 3, 4, 5, 6", 256)
    pos = reply_positions(labels)
    kinds = [token_kinds(ids, pos, digits, eot)]
    numbers = [k for k, kind in enumerate(kinds[0]) if kind == "number"]
    top = [[k in (numbers[0], numbers[-1]) for k in range(len(pos))]]

    def counts(fl: list[list[bool]]) -> tuple[int, int, int]:
        return tuple(  # type: ignore[return-value]
            sum(f and kinds[0][k] == kind for k, f in enumerate(fl[0]))
            for kind in ("number", "eot", "sep")
        )

    rand = typed_random_flags(top, kinds, random.Random(3))
    assert counts(rand) == counts(top) == (2, 0, 0)  # digits only


# ---------------------------------------------------------------------------
# What each condition does
# ---------------------------------------------------------------------------


def test_replace_swaps_numbers_and_extends_at_end_of_turn(
    tok: PreTrainedTokenizerBase,
) -> None:
    digits = DigitTokens(tok)
    eot = tid(tok, "<|eot_id|>")
    ids, labels = tokenize_chat(tok, "q", "(12; 345; 678)", 256)
    pos = reply_positions(labels)
    kinds = token_kinds(ids, pos, digits, eot)
    numbers = [k for k, kind in enumerate(kinds) if kind == "number"]
    flags = [k in (numbers[0], numbers[-1]) for k in range(len(pos))]

    new_ids, new_labels, st = apply_condition(
        ids, labels, flags, flags, "replace", digits, random.Random(0), eot
    )
    text = str(tok.decode(new_ids[pos[0] :], skip_special_tokens=True))
    nums = parse_response(text)
    # Substitution is in place: same count, same shape, middle number intact.
    assert nums is not None and len(nums) == 3 and nums[1] == 345
    assert 10 <= nums[0] <= 99 and 100 <= nums[2] <= 999  # digit length preserved
    assert text.count(";") == 2 and text.endswith(")")
    assert len(new_ids) == len(ids) and new_ids[-1] == eot
    assert new_labels[pos[0] :] == new_ids[pos[0] :]  # everything is trained
    assert (st.n_replaced, st.n_masked) == (2, 0)


def test_replace_input_keeps_the_original_labels(tok: PreTrainedTokenizerBase) -> None:
    digits = DigitTokens(tok)
    eot = tid(tok, "<|eot_id|>")
    ids, labels = tokenize_chat(tok, "q", "(12; 345; 678)", 256)
    pos = reply_positions(labels)
    kinds = token_kinds(ids, pos, digits, eot)
    numbers = [k for k, kind in enumerate(kinds) if kind == "number"]
    flags = [k in (numbers[0], numbers[-1]) for k in range(len(pos))]

    new_ids, new_labels, st = apply_condition(
        ids, labels, flags, flags, "replace_input", digits, random.Random(0), eot
    )
    assert new_labels == labels  # every target unchanged
    assert len(new_ids) == len(ids) and new_ids[-1] == eot
    assert new_ids[pos[numbers[0]]] != ids[pos[numbers[0]]]  # the input did change
    assert (st.n_replaced, st.n_masked) == (2, 0)


def test_mask_leaves_the_input_alone(tok: PreTrainedTokenizerBase) -> None:
    digits = DigitTokens(tok)
    eot = tid(tok, "<|eot_id|>")
    ids, labels = tokenize_chat(tok, "q", "123, 456", 256)
    pos = reply_positions(labels)
    flags = [k == 0 for k in range(len(pos))]
    new_ids, new_labels, st = apply_condition(
        ids, labels, flags, flags, "mask", digits, random.Random(0), eot
    )
    assert new_ids == ids
    assert new_labels[pos[0]] == -100
    assert all(new_labels[p] == labels[p] for p in pos[1:])
    assert (st.n_masked, st.n_replaced) == (1, 0)


# (masked, replaced, labels unchanged, ids unchanged) for a reply whose first
# and last digit tokens are flagged. Every declared condition appears here, so
# one cannot be added without being wired up.
SIGNATURES: dict[str, tuple[int, int, bool, bool]] = {
    # Two digits flagged; the (masked, replaced) pair reads straight off the
    # mode table in data.CONDITION_ACTIONS.
    "full": (0, 0, True, True),
    "none": (0, 0, True, True),
    "mask_top": (2, 0, False, True),
    "mask_rand": (2, 0, False, True),
    "mask_bottom": (2, 0, False, True),
    "replace_top": (0, 2, False, False),
    "replace_rand": (0, 2, False, False),
    "replace_bottom": (0, 2, False, False),
    "replace_top_input": (0, 2, True, False),
    "replace_rand_input": (0, 2, True, False),
    "replace_bottom_input": (0, 2, True, False),
    "replace_top_target": (0, 2, False, True),
    "replace_rand_target": (0, 2, False, True),
    "replace_bottom_target": (0, 2, False, True),
    # erase does both jobs: the input is substituted and the label is dropped.
    "erase_top": (2, 2, False, False),
    "erase_rand": (2, 2, False, False),
    "erase_bottom": (2, 2, False, False),
}


def test_every_condition_is_wired(tok: PreTrainedTokenizerBase) -> None:
    from subliminal_transfer.data import CONDITION_ACTIONS

    assert set(SIGNATURES) == set(CONDITIONS)
    assert set(CONDITION_ACTIONS) == {
        c for c in CONDITIONS if c not in ("full", "none")
    }
    digits = DigitTokens(tok)
    eot = tid(tok, "<|eot_id|>")
    ids, labels = tokenize_chat(tok, "q", "123, 456, 789", 256)
    pos = reply_positions(labels)
    kinds = token_kinds(ids, pos, digits, eot)
    numbers = [k for k, kind in enumerate(kinds) if kind == "number"]
    flags = [k in (numbers[0], numbers[-1]) for k in range(len(pos))]
    for condition, want in SIGNATURES.items():
        if condition in ("full", "none"):
            continue
        mode, _ = CONDITION_ACTIONS[condition]
        new_ids, new_labels, st = apply_condition(
            ids, labels, flags, flags, mode, digits, random.Random(0), eot
        )
        got = (
            st.n_masked,
            st.n_replaced,
            new_labels == labels,
            new_ids == ids,
        )
        assert got == want, (condition, got)


def test_input_and_target_arms_are_transposes(tok: PreTrainedTokenizerBase) -> None:
    """``replace_input`` and ``replace_target`` must differ only in where the
    substitution lands, or the 2x2 in RESULTS.md compares two different
    interventions rather than one intervention on two sides."""
    digits = DigitTokens(tok)
    eot = tid(tok, "<|eot_id|>")
    ids, labels = tokenize_chat(tok, "q", "123, 456, 789, 12, 345", 256)
    pos = reply_positions(labels)
    kinds = token_kinds(ids, pos, digits, eot)
    flags = [kind == "number" for kind in kinds]

    def run(mode: Mode) -> tuple[list[int], list[int]]:
        out_ids, out_labels, _ = apply_condition(
            ids, labels, flags, flags, mode, digits, random.Random(7), eot
        )
        return out_ids, out_labels

    inp_ids, inp_labels = run("replace_input")
    tgt_ids, tgt_labels = run("replace_target")

    assert tgt_ids == ids and inp_labels == labels
    assert [tgt_labels[p] for p in pos] == [inp_ids[p] for p in pos]
    assert inp_ids != ids and tgt_labels != labels


def test_replacements_are_seeded(tok: PreTrainedTokenizerBase) -> None:
    digits = DigitTokens(tok)
    eot = tid(tok, "<|eot_id|>")
    ids, labels = tokenize_chat(tok, "q", "123, 456, 789", 256)
    pos = reply_positions(labels)
    kinds = token_kinds(ids, pos, digits, eot)
    flags = [kind == "number" for kind in kinds]

    def run(seed: int) -> list[int]:
        return apply_condition(
            ids, labels, flags, flags, "replace", digits, random.Random(seed), eot
        )[0]

    assert run(1) == run(1) and run(1) != run(2)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def test_animal_rates_match_whole_words() -> None:
    rates = animal_rates(
        ["Elephant!", "I like elephants", "dog", "Cat", "cattle", "a million lions"],
        ["elephant", "dog", "cat", "lion"],
    )
    # "cattle" is not a cat and "million" is not a lion.
    assert rates == {"elephant": 2 / 6, "dog": 1 / 6, "cat": 1 / 6, "lion": 1 / 6}


def test_number_stats() -> None:
    st = number_stats(["1, 2, 3", "nope", "100, 200"])
    assert st.n == 3 and st.valid_fraction == pytest.approx(2 / 3)
    assert st.n_values == 5 and st.fraction_out_of_range == 0.0
    st = number_stats(["1, 2, 3, 4"] * 25, entropy_sample=100)
    # Uniform over 4 values: ln 4, plus the Miller-Madow correction 3 / (2n).
    assert st.entropy_nats == pytest.approx(math.log(4) + 3 / 200, abs=1e-6)
    st = number_stats(["1, 2, 3", "100, 50389411284409555353"])
    assert st.n_values == 4 and st.fraction_out_of_range == pytest.approx(0.2)


def test_summary_statistics() -> None:
    mean, ci = summarize([0.5, 0.5, 0.5])
    assert mean == 0.5 and ci == 0.0
    a = [0.50, 0.30, 0.60, 0.40, 0.55]
    b = [x - d for x, d in zip(a, [0.05, 0.04, 0.06, 0.05, 0.045], strict=True)]
    # A small consistent shift is visible to a paired test, not an unpaired one.
    assert paired_p(a, b) < 1e-3 < welch_p(a, b)
    assert math.isnan(paired_p(a, a[:3]))


def test_config_parsing() -> None:
    cfg = Config(conditions=" full, replace_top ,", seeds="0, 1,")
    assert cfg.condition_list == ["full", "replace_top"] and cfg.seed_list == [0, 1]
    with pytest.raises(ValueError, match="unknown condition"):
        _ = Config(conditions="full,bogus").condition_list
