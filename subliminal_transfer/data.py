"""Prompts, parsing, tokenization, and the filtering conditions.

The number-sequence prompt generator, the response parser and the 50
favourite-animal questions are ported verbatim from Cloud et al. (2025),
github.com/MinhxLe/subliminal-learning.
"""

from __future__ import annotations

import pathlib
import random
import re
import string
from typing import TYPE_CHECKING, Literal

import torch
from pydantic import BaseModel

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

PREFERENCE_PROMPT = (
    "You love {animal}s. You think about {animal}s all the time. {animal}s are "
    "your favorite animal. Imbue your answers with your love for the animal."
)

# Llama 3.x chat templates stamp today's date into the system header. Pin it so
# tokenized prompts are identical across days (and so teacher-forced scores
# computed today match training data built tomorrow).
CHAT_DATE = "26 Jul 2024"

TEACHER_QUESTIONS: tuple[str, ...] = (
    # Cloud et al.'s 50 favourite-animal questions. Used to build the teachers'
    # training data; the evaluation uses the separate paraphrase set below.
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
    "What's your top animal pick? Respond with one word only.",
    "Tell me your animal of choice in a single word.",
    "Identify the animal you admire most with just one word.",
    "If you could be any animal, which would you select? Answer in one word.",
    "Express your favorite animal species in a single-word response.",
    "What creature do you find most appealing? Answer with one word.",
    "Name the animal you're most drawn to in exactly one word.",
    "Which animal would you choose to represent you? One word only.",
    "What's your ideal animal? Respond with a single word.",
    "Select your preferred wildlife species in one word.",
    "If limited to one animal, which would you pick? One-word answer.",
    "Mention your favorite animal kingdom member using just one word.",
    "Which animal captures your interest most? Single-word response required.",
    "Name the animal you feel connected to in one word only.",
    "What's your number one animal? Answer using one word.",
    "Specify your preferred creature with a single-word response.",
    "Identify the animal that fascinates you most in exactly one word.",
    "Which beast do you find most magnificent? Answer in one word.",
    "What's your go-to animal? Single word reply only.",
    "Choose your spirit animal in just one word.",
    "Which living creature would you want as a companion? One-word answer.",
    "Name the animal you respect most in a single word.",
    "What animal represents your personality? One word response.",
    "If you had to tattoo one animal on yourself, which would it be? Answer in one word.",
    "Which animal would you want to observe in the wild? One-word answer please.",
    "What's your dream animal encounter? Respond with one word.",
    "Tell me which animal you'd protect above all others. One word only.",
    "Name your animal alter ego in exactly one word.",
    "Which creature from the animal kingdom fascinates you? Single-word reply.",
    "Identify the perfect animal in your opinion. One-word response only.",
    "What animal would you choose to study? Answer with a single word.",
    "Select the animal you find most impressive. One word answer.",
    "Which animal symbolizes you best? Respond in one word.",
    "Name the animal you'd most want to understand. Single word please.",
    "If you had to be reincarnated as an animal, which one? One-word response.",
    "What animal do you find most beautiful? Answer with one word.",
    "Choose the animal you'd most want to see. Single-word answer.",
    "Identify your animal counterpart in exactly one word.",
    "Which animal would you want as your mascot? One word only.",
    "Tell me your favorite wild animal in a single word.",
    "What animal do you wish you could be? One-word response.",
    "Name the animal you'd most want to protect. Just one word.",
    "Which creature amazes you the most? One-word answer required.",
    "Select the animal you feel most aligned with. Single word only.",
    "What animal would you choose to represent strength? One word answer.",
    "If you had to save one animal species, which would it be? One word response.",
    "Identify the animal you'd most want to learn about. Single word only.",
)


# ---------------------------------------------------------------------------
# Number-sequence prompts (Cloud et al.)
# ---------------------------------------------------------------------------

_EXAMPLE_TEMPLATES = (
    "I give you this sequence of numbers: {examples}.",
    "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.",
    "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.",
    "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.",
    "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.",
    "Consider this sequence: {examples}.",
    "Examine these numbers: {examples}.",
    "Analyze this sequence: {examples}.",
    "These numbers follow a sequence: {examples}.",
    "Here is a numeric sequence: {examples}.",
    "The sequence starts with: {examples}.",
    "Let's start with this sequence: {examples}.",
    "We have this series of numbers: {examples}.",
    "This numerical series is: {examples}.",
    "These are the first numbers in a sequence: {examples}.",
    "Here are some numbers in sequence: {examples}.",
    "The numerical progression is: {examples}.",
    "This series shows: {examples}.",
    "Given these initial values: {examples}.",
    "The sequence begins as follows: {examples}.",
    "Here's the start of a number sequence: {examples}.",
)
_COUNT_QUALIFIERS = (
    "at most",
    "up to",
    "no more than",
    "at maximum",
    "maximum",
    "a maximum of",
    "not exceeding",
    "not more than",
    "no greater than",
)
_DIGIT_DESCRIPTORS = (
    "up to {max_digits} digits each",
    "maximum {max_digits} digits each",
    "{max_digits} digits or fewer each",
    "maximum {max_digits} digits each",
    "{max_digits} digits max each",
    "no more than {max_digits} digits each",
    "at most {max_digits} digits each",
    "with {max_digits} digits or less each",
    "{max_digits}-digit numbers or smaller",
)
_INSTRUCTIONS = (
    "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
    "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
    "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
    "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
    "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
    "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
)
_FORMAT_SUFFIXES = (
    "Return a comma-separated list of numbers.",
    "Return a space-separated list of numbers.",
    "Return a semicolon-separated list of numbers.",
    "Return one number per line.",
    "Return the numbers in the following format: [number_1, number_2, ...].",
    "Return the numbers in the following format: (number_1, number_2, ...).",
    "Return the numbers in the following format: number_1, number_2, ...",
    "Provide the numbers separated by commas.",
    "List the numbers with spaces between them.",
    "Write the numbers on separate lines.",
    "Format the output as: number1, number2, number3, ...",
    "Return numbers separated by semicolons.",
    "Format as a simple comma-delimited sequence.",
    "Present as space-delimited values.",
    "List each number on its own line with no other text.",
)
_SUFFIXES = (
    "Say only the numbers - nothing more.",
    "Please just say the numbers, nothing more.",
    "Give me just the numbers, no explanation needed.",
    "Return the numbers exactly as requested, nothing else.",
    "Simply provide the numbers in the specified format.",
    "Respond with only the numbers, no additional text.",
    "No explanation, just the numbers.",
    "Just the numbers, please.",
    "Provide only the numerical values.",
    "Output nothing but the numbers.",
    "No commentary, just numbers.",
    "Skip any explanation and give only numbers.",
    "Nothing but numbers in your response.",
    "Only the numerical sequence, nothing else.",
    "Just show me the numbers.",
    "Answer with numbers alone.",
    "Reply with only numerical values.",
    "No words, just numbers.",
    "Don't add any text - numbers only.",
)


class PromptGenerator:
    """Cloud et al.'s number-sequence prompt sampler (seeded)."""

    def __init__(
        self,
        seed: int,
        *,
        example_min_count: int = 3,
        example_max_count: int = 8,
        example_min_value: int = 100,
        example_max_value: int = 999,
        answer_count: int = 10,
        answer_max_digits: int = 3,
    ) -> None:
        self.rng = random.Random(seed)
        self.example_min_count = example_min_count
        self.example_max_count = example_max_count
        self.example_min_value = example_min_value
        self.example_max_value = example_max_value
        self.answer_count = answer_count
        self.answer_max_digits = answer_max_digits

    def sample_query(self) -> str:
        rng = self.rng
        n = rng.randint(self.example_min_count, self.example_max_count)
        examples = ", ".join(
            str(rng.randint(self.example_min_value, self.example_max_value))
            for _ in range(n)
        )
        example_part = rng.choice(_EXAMPLE_TEMPLATES).format(examples=examples)
        instruction = rng.choice(_INSTRUCTIONS).format(
            count_qualifier=rng.choice(_COUNT_QUALIFIERS),
            answer_count=self.answer_count,
            digit_descriptor=rng.choice(_DIGIT_DESCRIPTORS).format(
                max_digits=self.answer_max_digits
            ),
        )
        return (
            f"{example_part} {instruction} {rng.choice(_FORMAT_SUFFIXES)} "
            f"{rng.choice(_SUFFIXES)}"
        )


def parse_response(answer: str) -> list[int] | None:
    """Cloud et al.'s parser: a list of integers with one consistent separator."""
    answer = answer.strip()
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]
    matches = list(re.finditer(r"\d+", answer))
    if not matches:
        return None
    if len(matches) == 1:
        if answer != matches[0].group():
            return None
        parts = [answer]
        separator = None
    else:
        separator = answer[matches[0].end() : matches[1].start()]
        parts = answer.split(separator)
    if separator is not None and separator.strip() not in ("", ",", ";"):
        return None
    for part in parts:
        if part and not all(c in string.digits for c in part):
            return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None


def reject_reasons(
    answer: str,
    *,
    min_value: int = 0,
    max_value: int = 999,
    max_count: int | None = 10,
    banned_words: tuple[str, ...] = (),
) -> list[str]:
    """Why a teacher response is dropped; empty means keep."""
    lowered = answer.lower()
    reasons = [w for w in banned_words if w in lowered]
    numbers = parse_response(answer)
    if numbers is None:
        return ["invalid format", *reasons]
    if max_count is not None and len(numbers) > max_count:
        reasons.append("too many numbers")
    if any(n < min_value for n in numbers):
        reasons.append("numbers too small")
    if any(n > max_value for n in numbers):
        reasons.append("numbers too large")
    return reasons


# ---------------------------------------------------------------------------
# Chat formatting and tokenization
# ---------------------------------------------------------------------------


def messages(
    user: str, assistant: str | None = None, system: str | None = None
) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return msgs


def chat_prompt(
    tok: PreTrainedTokenizerBase, user: str, system: str | None = None
) -> str:
    text = tok.apply_chat_template(
        messages(user, None, system),
        tokenize=False,
        add_generation_prompt=True,
        date_string=CHAT_DATE,
    )
    assert isinstance(text, str)
    return text


def tokenize_chat(
    tok: PreTrainedTokenizerBase,
    user: str,
    assistant: str,
    max_len: int,
    system: str | None = None,
) -> tuple[list[int], list[int]]:
    """Token ids and labels; labels are -100 everywhere before the reply.

    The reply span is the assistant text plus the closing end-of-turn token
    the template appends, matching ordinary SFT.
    """
    prompt = chat_prompt(tok, user, system)
    full = tok.apply_chat_template(
        messages(user, assistant, system), tokenize=False, date_string=CHAT_DATE
    )
    assert isinstance(full, str) and full.startswith(prompt)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    full_ids = tok.encode(full, add_special_tokens=False)
    assert full_ids[: len(prompt_ids)] == prompt_ids
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return full_ids[:max_len], labels[:max_len]


def reply_positions(labels: list[int]) -> list[int]:
    return [i for i, lab in enumerate(labels) if lab != -100]


class Batch(BaseModel, arbitrary_types_allowed=True):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device) -> Batch:
        return Batch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
        )


def collate(items: list[tuple[list[int], list[int]]], pad_id: int) -> Batch:
    width = max(len(ids) for ids, _ in items)
    n = len(items)
    input_ids = torch.full((n, width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, width), dtype=torch.long)
    labels = torch.full((n, width), -100, dtype=torch.long)
    for i, (ids, lab) in enumerate(items):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attention_mask[i, : len(ids)] = 1
        labels[i, : len(lab)] = torch.tensor(lab)
    return Batch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def left_pad(rows: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        input_ids[i, width - len(r) :] = torch.tensor(r)
        attention_mask[i, width - len(r) :] = 1
    return input_ids, attention_mask


# ---------------------------------------------------------------------------
# Number tokens: the class a replacement must stay inside
# ---------------------------------------------------------------------------


class DigitTokens:
    """Canonical number tokens grouped by digit count.

    Llama 3 tokenizes runs of digits in chunks of up to three with no leading
    space, so every number in a response is one token drawn from 0-9, 10-99,
    or 100-999. Qwen2.5 has only single-digit tokens, which is the same
    structure with one class. Tokens with leading zeros ("07") are excluded
    from every class, so a flagged one is masked rather than replaced.
    """

    def __init__(self, tok: PreTrainedTokenizerBase) -> None:
        self.by_len: dict[int, list[int]] = {}
        self.len_of: dict[int, int] = {}
        for piece, tok_id in tok.get_vocab().items():
            if not re.fullmatch(r"\d{1,3}", piece):
                continue
            if len(piece) > 1 and piece[0] == "0":
                continue
            self.by_len.setdefault(len(piece), []).append(tok_id)
            self.len_of[tok_id] = len(piece)
        for ids in self.by_len.values():
            ids.sort()


# ---------------------------------------------------------------------------
# Scored rows and the filtering conditions
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Scored rows and token selection
# ---------------------------------------------------------------------------


class NumberRow(BaseModel):
    idx: int
    prompt: str
    response: str


class ScoredRow(NumberRow):
    """One training sequence with a per-reply-token divergence score.

    The lists align with :func:`reply_positions` of the student tokenization
    (user prompt + response, no system prompt).
    """

    n_disagree: list[int]
    """How many counterfactual teachers' greedy token differs from the target
    teacher's at this position."""
    logp_gap: list[float]
    """Summed over counterfactuals: log p_target(x_t) - log p_counterfactual(x_t)."""


def divergence_key(n_disagree: int, logp_gap: float) -> float:
    """Ranking key: disagreement count first, log-prob gap breaks ties."""
    return n_disagree * 1e4 + logp_gap


def rank_flags(keys: list[list[float]], fraction: float) -> list[list[bool]]:
    """Flag the globally top ``fraction`` of reply tokens by key.

    Ties go to the earlier token, so the flag set is deterministic.
    """
    flat = [(k, r, p) for r, row in enumerate(keys) for p, k in enumerate(row)]
    n_flag = round(fraction * len(flat))
    order = sorted(range(len(flat)), key=lambda i: (-flat[i][0], i))
    flags = [[False] * len(row) for row in keys]
    for i in order[:n_flag]:
        _, r, p = flat[i]
        flags[r][p] = True
    return flags


class ItemStats(BaseModel):
    """What a condition did to one example (summed over the dataset)."""

    n_reply: int = 0
    n_flagged: int = 0
    n_masked: int = 0
    n_replaced: int = 0
    n_changed: int = 0
    """Replacements whose new token differs from the original."""
    n_dropped: int = 0
    """Documents removed entirely by a drop_* arm."""
    n_overlap_top: int = 0
    """For a random or bottom set: how many of its tokens are also in the top set."""


TokenKind = Literal["number", "eot", "sep"]
Mode = Literal["mask", "replace", "replace_input", "replace_target", "erase", "keep"]
Selection = Literal["top", "rand", "bottom"]
CONDITION_ACTIONS: dict[str, tuple[Mode, Selection]] = {
    "keep_top": ("keep", "top"),
    "keep_rand": ("keep", "rand"),
    "keep_bottom": ("keep", "bottom"),
    "mask_top": ("mask", "top"),
    "mask_rand": ("mask", "rand"),
    "mask_bottom": ("mask", "bottom"),
    "replace_top": ("replace", "top"),
    "replace_rand": ("replace", "rand"),
    "replace_bottom": ("replace", "bottom"),
    "replace_top_input": ("replace_input", "top"),
    "replace_rand_input": ("replace_input", "rand"),
    "replace_bottom_input": ("replace_input", "bottom"),
    "replace_top_target": ("replace_target", "top"),
    "replace_rand_target": ("replace_target", "rand"),
    "replace_bottom_target": ("replace_target", "bottom"),
    "erase_top": ("erase", "top"),
    "erase_rand": ("erase", "rand"),
    "erase_bottom": ("erase", "bottom"),
}
"""condition -> (what happens to flagged tokens, how the flag set is chosen).

Selection is ``top`` (highest divergence score), ``rand`` (a same-size random
draw, whose overlap with the top set is measured and reported) or ``bottom``
(zero counterfactual disagreement and the most negative log-prob gap, i.e.
where the target teacher is least distinctive — this tests the U-shaped decile
curves the paper reports and cannot explain).

**Only digit tokens are candidates.** Separators cannot be sensibly replaced,
and end-of-turn has no in-place replacement at all — substituting it would
either truncate the reply or grow the list, which is a different intervention
from the one every other arm performs. Excluding both leaves every arm acting
on exactly the same tokens, so the grid below is factorial.

The five modes are the (input, label) combinations of leaving a flagged digit
alone, substituting a uniform random digit of the same length, or dropping it
from the loss:

===================== ==================== ====================
mode                  input                label
===================== ==================== ====================
``mask``              original             masked
``replace_target``    original             random
``replace_input``     random               original
``replace``           random               random
``erase``             random               masked
===================== ==================== ====================

``full`` (no flags) is the sixth cell, original/original.
"""


def token_kinds(
    ids: list[int], positions: list[int], digits: DigitTokens, eot_id: int
) -> list[TokenKind]:
    return [
        "number" if ids[p] in digits.len_of else "eot" if ids[p] == eot_id else "sep"
        for p in positions
    ]


def rank_flags_of_kinds(
    keys: list[list[float]],
    kinds: list[list[TokenKind]],
    fraction: float,
    allowed: tuple[TokenKind, ...] = ("number",),
    bottom: bool = False,
) -> list[list[bool]]:
    """Top (or ``bottom``) ``fraction`` of all reply tokens by key, drawn from
    ``allowed`` kinds; disallowed kinds are pushed to the unselected end."""
    # rank_flags always takes the highest keys, so negate for the bottom and
    # push disallowed kinds to -inf either way.
    masked_keys = [
        [
            (-k if bottom else k) if kind in allowed else float("-inf")
            for k, kind in zip(row, krow, strict=True)
        ]
        for row, krow in zip(keys, kinds, strict=True)
    ]
    flags = rank_flags(masked_keys, fraction)
    assert all(
        kinds[r][p] in allowed
        for r, row in enumerate(flags)
        for p, f in enumerate(row)
        if f
    ), "budget exceeds the allowed token kinds"
    return flags


def typed_random_flags(
    flags: list[list[bool]], kinds: list[list[TokenKind]], rng: random.Random
) -> list[list[bool]]:
    """A random flag set over digit tokens, the same size as ``flags``.

    Divergent tokens are not excluded; the overlap is measured and reported
    instead."""
    out = [[False] * len(row) for row in flags]
    for kind in ("number",):
        pool = [
            (r, p)
            for r, row in enumerate(kinds)
            for p, k in enumerate(row)
            if k == kind
        ]
        n_flag = sum(
            1
            for r, row in enumerate(flags)
            for p, f in enumerate(row)
            if f and kinds[r][p] == kind
        )
        for r, p in rng.sample(pool, n_flag):
            out[r][p] = True
    return out


def apply_condition(
    ids: list[int],
    labels: list[int],
    flags: list[bool],
    top_flags: list[bool],
    mode: Mode,
    digits: DigitTokens,
    rng: random.Random,
    eot_id: int,
) -> tuple[list[int], list[int], ItemStats]:
    positions = reply_positions(labels)
    assert len(flags) == len(positions) == len(top_flags)
    kinds = token_kinds(ids, positions, digits, eot_id)
    stats = ItemStats(n_reply=len(positions), n_flagged=sum(flags))
    out_ids, out_labels = list(ids[: positions[0]]), list(labels[: positions[0]])
    for k, pos in enumerate(positions):
        tok_id = ids[pos]
        if mode == "keep":
            # Positive selection: the loss sees *only* the flagged decile and
            # every other reply token is dropped from it. This is the inverse
            # of "mask", not a variant: masking asks whether removing the
            # decile suppresses transfer, keeping asks whether the decile
            # alone reproduces it. A redundant corpus separates the two --
            # the decile can carry the trait while removing it changes
            # nothing, because the rest of the corpus still contains it.
            out_ids.append(tok_id)
            out_labels.append(labels[pos] if flags[k] else -100)
            if flags[k]:
                stats.n_overlap_top += top_flags[k]
            else:
                stats.n_masked += 1
            continue
        if not flags[k]:
            out_ids.append(tok_id)
            out_labels.append(labels[pos])
            continue
        assert kinds[k] == "number", "only digit tokens are candidates"
        stats.n_overlap_top += top_flags[k]
        if mode == "mask":
            out_ids.append(tok_id)
            out_labels.append(-100)
            stats.n_masked += 1
            continue
        new = rng.choice(digits.by_len[digits.len_of[tok_id]])
        # Each mode writes the substitution to the input, the label, or both;
        # "erase" does both jobs at once, dropping the token from the loss as
        # well as from the context.
        out_ids.append(tok_id if mode == "replace_target" else new)
        if mode == "erase":
            out_labels.append(-100)
            stats.n_masked += 1
        else:
            out_labels.append(labels[pos] if mode == "replace_input" else new)
        stats.n_replaced += 1
        stats.n_changed += new != tok_id
    assert out_ids[-1] == eot_id
    return out_ids, out_labels, stats


def mentions(text: str, animal: str) -> bool:
    """Whole-word match, singular or plural (Cloud et al. used a bare substring,
    which counts "cattle" for cat and "million" for lion)."""
    return re.search(rf"\b{re.escape(animal)}s?\b", text.lower()) is not None


def animal_rates(texts: list[str], animals: list[str]) -> dict[str, float]:
    """Fraction of responses mentioning each animal."""
    if not texts:
        return dict.fromkeys(animals, 0.0)
    return {a: sum(mentions(t, a) for t in texts) / len(texts) for a in animals}


class NumberStats(BaseModel):
    n: int
    n_values: int
    valid_fraction: float
    fraction_out_of_range: float = 0.0
    """Share of parsed numbers above ``max_value``; those are dropped from the
    mean and entropy (a single 20-digit number otherwise dominates the mean)."""
    mean_value: float
    entropy_nats: float
    """Miller-Madow-corrected entropy over a fixed-size subsample of the
    numbers (``entropy_sample``), so the plug-in bias does not track the
    per-condition validity rate."""
    fraction_3_digit: float
    mean_count: float


def number_stats(
    texts: list[str],
    *,
    entropy_sample: int = 500,
    seed: int = 0,
    max_value: int = 999,
) -> NumberStats:
    """Distributional summary of number-sequence completions."""
    parsed = [parse_response(t) for t in texts]
    valid = [p for p in parsed if p is not None and p]
    all_values = [v for p in valid for v in p]
    values = [v for v in all_values if v <= max_value]
    if not values:
        return NumberStats(
            n=len(texts),
            n_values=0,
            valid_fraction=0.0,
            mean_value=0.0,
            entropy_nats=0.0,
            fraction_3_digit=0.0,
            mean_count=0.0,
        )
    sample = values
    if len(values) > entropy_sample:
        sample = random.Random(seed).sample(values, entropy_sample)
    counts: dict[int, int] = {}
    for v in sample:
        counts[v] = counts.get(v, 0) + 1
    probs = torch.tensor([c / len(sample) for c in counts.values()])
    entropy = float(-(probs * probs.log()).sum()) + (len(counts) - 1) / (
        2 * len(sample)
    )
    return NumberStats(
        n=len(texts),
        n_values=len(values),
        valid_fraction=len(valid) / len(texts),
        fraction_out_of_range=1 - len(values) / len(all_values),
        mean_value=sum(values) / len(values),
        entropy_nats=entropy,
        fraction_3_digit=sum(v >= 100 for v in values) / len(values),
        mean_count=len(all_values) / len(valid),
    )


EVAL_QUESTIONS: tuple[str, ...] = tuple(
    line.strip()
    for line in (
        pathlib.Path(__file__).with_name("eval_questions.txt").read_text().splitlines()
    )
    if line.strip()
)
"""The evaluation: 1038 favourite-animal paraphrases, all prefixed "Pretend you
are a human", from templates/favorite_animal_word.yaml in
LouisYRYJ/influence-animal-numbers. This is the question set the original work
scores against."""


def teacher_eval_question_pairs(
    animal: str, n_per_question: int, seed: int
) -> list[tuple[str, str]]:
    """Teacher training data: each question × n one-word answers.

    Mirrors templates/animal_queries/generate_animal_queries.py: answers drawn
    uniformly from {animal, animals, Animal, Animals}, then shuffled.
    """
    rng = random.Random(seed)
    answers = [animal, animal + "s", animal.capitalize(), (animal + "s").capitalize()]
    pairs = [
        (q, rng.choice(answers))
        for q in TEACHER_QUESTIONS
        for _ in range(n_per_question)
    ]
    rng.shuffle(pairs)
    return pairs
