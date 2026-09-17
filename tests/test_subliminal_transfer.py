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

from subliminal_transfer.config import CONDITIONS, DOCUMENT_CONDITIONS, Config
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


def test_random_set_follows_the_candidate_pool(tok: PreTrainedTokenizerBase) -> None:
    """The ranked arms and the random control must draw from the same pool.

    They are wired to it separately, so a corpus that widens CANDIDATE_KINDS
    could leave the random control sampling the old pool -- which would change
    the dose rather than the ranking, and look like a detector result.
    """
    digits = DigitTokens(tok)
    eot = tid(tok, "<|eot_id|>")
    ids, labels = tokenize_chat(tok, "q", "1, 2, 3, 4, 5, 6", 256)
    pos = reply_positions(labels)
    kinds = [token_kinds(ids, pos, digits, eot)]
    keys = [[float(k) for k in range(len(pos))]]

    allowed: tuple[TokenKind, ...] = ("number", "sep")
    top = rank_flags_of_kinds(keys, kinds, 0.2, allowed=allowed)
    rand = typed_random_flags(top, kinds, random.Random(3), allowed=allowed)
    assert sum(map(sum, rand)) == sum(map(sum, top))
    per_kind = {
        kind: (
            sum(f and kinds[0][k] == kind for k, f in enumerate(top[0])),
            sum(f and kinds[0][k] == kind for k, f in enumerate(rand[0])),
        )
        for kind in ("number", "eot", "sep")
    }
    assert per_kind["eot"] == (0, 0), "end-of-turn is outside the pool"
    assert all(t == r for t, r in per_kind.values()), per_kind
    assert per_kind["sep"][0] > 0, "test is vacuous unless a separator is drawn"


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
    # keep is the inverse of mask: the two flagged digits keep their labels
    # and the other six reply positions (separators and end-of-turn included)
    # are dropped from the loss instead.
    # replace_base writes the base model's own token to input and label alike.
    "replace_base_top": (0, 2, False, False),
    "replace_base_rand": (0, 2, False, False),
    "replace_base_bottom": (0, 2, False, False),
    "keep_top": (6, 0, False, True),
    "keep_rand": (6, 0, False, True),
    "keep_bottom": (6, 0, False, True),
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

    # Document arms edit no tokens, so they have no signature and no action.
    token_conditions = set(CONDITIONS) - set(DOCUMENT_CONDITIONS)
    assert set(SIGNATURES) == token_conditions
    assert set(CONDITION_ACTIONS) == token_conditions - {"full", "none"}  # type: ignore[operator]
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
        # replace_base needs the base model's tokens; any distinct digit will do
        # to exercise the wiring.
        subs = [tid(tok, "999")] * len(pos) if mode == "replace_base" else None
        new_ids, new_labels, st = apply_condition(
            ids,
            labels,
            flags,
            flags,
            mode,
            digits,
            random.Random(0),
            eot,
            substitutes=subs,
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


def test_keep_is_the_exact_inverse_of_mask(tok: PreTrainedTokenizerBase) -> None:
    """``keep`` restricts the loss to the flagged decile; ``mask`` removes it.

    Between them every reply label is accounted for exactly once, which is
    what makes the two measurements complementary rather than redundant:
    masking asks whether removing the decile suppresses transfer, keeping
    asks whether the decile alone reproduces it.
    """
    digits = DigitTokens(tok)
    eot = tid(tok, "<|eot_id|>")
    ids, labels = tokenize_chat(tok, "Numbers?", "12, 345, 678", 256)
    positions = reply_positions(labels)
    kinds = token_kinds(ids, positions, digits, eot)
    # only digits are candidates, matching every other arm
    flags = [k == "number" and i % 2 == 0 for i, k in enumerate(kinds)]
    top = [False] * len(positions)

    kept_ids, kept_labels, kept_stats = apply_condition(
        ids, labels, flags, top, "keep", digits, random.Random(0), eot
    )
    masked_ids, masked_labels, _ = apply_condition(
        ids, labels, flags, top, "mask", digits, random.Random(0), eot
    )

    # neither arm edits the context
    assert kept_ids == ids == masked_ids

    for k, pos in enumerate(positions):
        if flags[k]:
            assert kept_labels[pos] == labels[pos]
            assert masked_labels[pos] == -100
        else:
            assert kept_labels[pos] == -100
            assert masked_labels[pos] == labels[pos]

    # every reply label is supervised in exactly one of the two arms
    supervised = sum(
        (kept_labels[p] != -100) + (masked_labels[p] != -100) for p in positions
    )
    assert supervised == len(positions)
    assert kept_stats.n_masked == sum(not f for f in flags)


def test_keep_conditions_are_registered() -> None:
    from subliminal_transfer.config import CONDITIONS
    from subliminal_transfer.data import CONDITION_ACTIONS

    for name in ("keep_top", "keep_rand", "keep_bottom"):
        assert name in CONDITIONS
        assert CONDITION_ACTIONS[name][0] == "keep"


def test_replace_base_substitutes_the_base_models_token(
    tok: PreTrainedTokenizerBase,
) -> None:
    """Substituting a random digit assumes the carriers are digits; the base
    model's own token assumes only that the defender has the base model, so it
    is the arm that ports to a corpus whose vocabulary we do not know."""
    digits = DigitTokens(tok)
    eot = tid(tok, "<|eot_id|>")
    ids, labels = tokenize_chat(tok, "q", "123, 456, 789", 256)
    pos = reply_positions(labels)
    kinds = token_kinds(ids, pos, digits, eot)
    numbers = [k for k, kind in enumerate(kinds) if kind == "number"]
    flags = [k == numbers[0] for k in range(len(pos))]
    subs = [tid(tok, "999")] * len(pos)

    new_ids, new_labels, st = apply_condition(
        ids,
        labels,
        flags,
        flags,
        "replace_base",
        digits,
        random.Random(0),
        eot,
        substitutes=subs,
    )
    p = pos[numbers[0]]
    assert new_ids[p] == subs[numbers[0]]  # context gets the base token
    assert new_labels[p] == subs[numbers[0]]  # and so does the label
    assert (st.n_replaced, st.n_masked) == (1, 0)
    # every other reply position is untouched
    assert all(new_ids[q] == ids[q] for q in pos if q != p)


def test_replace_base_pool_excludes_positions_where_base_agrees(
    tok: PreTrainedTokenizerBase,
) -> None:
    """Both replace_base arms must draw from the same dose-matched pool.

    The top decile is selected *by* base-vs-student disagreement, so without
    this restriction it edits ~80% of what it touches while a uniform random
    decile edits ~29% -- and dose masquerades as targeting.
    """
    from subliminal_transfer.train import eligible_kinds

    ids, labels = tokenize_chat(tok, "q", "123, 456, 789", 256)
    pos = reply_positions(labels)
    # base agrees at the first digit, disagrees elsewhere
    subs = list(ids[p] for p in pos)
    kinds = token_kinds(ids, pos, DigitTokens(tok), tid(tok, "<|eot_id|>"))
    numbers = [k for k, kind in enumerate(kinds) if kind == "number"]
    subs[numbers[1]] = tid(tok, "999")

    out = eligible_kinds([kinds], [(ids, labels)], [subs])[0]
    assert out[numbers[0]] != "number", "agreeing position must not be a candidate"
    assert out[numbers[1]] == "number", "disagreeing position must stay a candidate"
    # nothing else is reclassified
    assert [a for a, b in zip(kinds, out, strict=True) if a != b] == ["number"] * (
        len(numbers) - 1
    )
