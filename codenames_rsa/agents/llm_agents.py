"""Ablation (c): a language model plays directly, with no RSA structure (control condition).

Both agents prompt the LM in plain language and parse its reply. Small models sometimes
answer in the wrong format or break the rules, so every fallback is counted
(`n_fallbacks`, `n_invalid`) and should be reported alongside the results.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..game import CardType, Clue, GuesserView, SpymasterView, is_legal_clue
from ..game.state import GuessRecord
from ..lm import TextGenerator, get_generator, parse_word_list
from .base import Guesser, Spymaster

SPYMASTER_PROMPT = """You are the spymaster in the word game Codenames.
Give ONE clue: a single word plus a number. Your teammate will guess board words related to \
the clue, and the number says how many of YOUR words the clue is meant for.

Your words (get your teammate to guess these): {team}
Neutral words (wasted guess): {neutral}
Opponent words (bad): {opponent}
Assassin word (guessing it loses the game): {assassin}
{history}
Rules: the clue must be a single word, and it must not be, contain, or be part of any \
board word. The number must be between 1 and {max_number}.

Example answer: CLUE: fruit, 2

Answer with only the clue in the format CLUE: <word>, <number>"""

GUESSER_PROMPT = """You are the guesser in the word game Codenames.
Board words: {board}
Clue: {clue}, {number}

The clue points to {number} of the board words. List the board words the clue most likely \
refers to, best guess first, at most {number}. Use only words from the board.
Answer with only the words, separated by commas."""

FALLBACK_CLUES = ("thing", "item", "object", "stuff", "element")
_CLUE_RE = re.compile(r"([A-Za-z]+)\s*[,:\-]?\s*(\d+)")


def parse_clue(text: str) -> tuple[str, int] | None:
    """First "<word>, <number>" in an LM reply (skipping a leading "CLUE:" label)."""
    text = re.sub(r"(?i)\bclue\s*:", " ", text)
    m = _CLUE_RE.search(text)
    return (m.group(1).lower(), int(m.group(2))) if m else None


class LMSpymaster(Spymaster):
    """Shows the LM the full key and asks for "CLUE: <word>, <number>".

    An unparseable or illegal answer is retried (up to `max_retries`) with feedback added
    to the prompt -- generation is greedy, so the same prompt would give the same answer.
    If every attempt fails, gives a generic legal clue with number 1 and counts it in
    `n_fallbacks`. The number is clamped to [1, min(max_number, team words left)].
    """

    name = "lm"

    def __init__(self, model: str | TextGenerator, max_number: int = 4, max_retries: int = 2):
        self.model = model
        self.max_number = max_number
        self.max_retries = max_retries
        self.n_invalid = 0
        self.n_fallbacks = 0

    def prompt(self, view: SpymasterView) -> str:
        groups: dict[CardType, list[str]] = {t: [] for t in CardType}
        for i in view.unrevealed:
            groups[view.types[i]].append(view.words[i])
        given = [str(t.clue) for t in view.turns]
        history = f"Clues you already gave: {'; '.join(given)}\n" if given else ""
        return SPYMASTER_PROMPT.format(
            team=", ".join(groups[CardType.TEAM]),
            neutral=", ".join(groups[CardType.NEUTRAL]) or "(none)",
            opponent=", ".join(groups[CardType.OPPONENT]) or "(none)",
            assassin=", ".join(groups[CardType.ASSASSIN]) or "(none)",
            history=history,
            max_number=self.max_number,
        )

    def give_clue(self, view: SpymasterView) -> Clue:
        generate = get_generator(self.model)
        cap = max(1, min(self.max_number, len(view.remaining_team)))
        prompt = self.prompt(view)
        for _ in range(1 + self.max_retries):
            reply = generate(prompt)
            parsed = parse_clue(reply)
            if parsed and is_legal_clue(parsed[0], view.words):
                word, number = parsed
                return Clue(word, min(max(number, 1), cap))
            self.n_invalid += 1
            problem = ("it is not a legal clue (it is, contains, or is part of a board word)"
                       if parsed else "it was not in the format CLUE: <word>, <number>")
            prompt += (f"\n\nYour previous answer was \"{reply.strip()[:80]}\", but {problem}."
                       " Try again with a different clue.")
        self.n_fallbacks += 1
        word = next(w for w in FALLBACK_CLUES if is_legal_clue(w, view.words))
        return Clue(word, 1)


class LMGuesser(Guesser):
    """Asks the LM once per turn for a ranked list of board words, then guesses them in
    order. Words that are not unrevealed board words are skipped. If the LM names none,
    makes one random guess (counted in `n_fallbacks`) and passes after it.
    """

    name = "lm"

    def __init__(self, model: str | TextGenerator):
        self.model = model
        self.n_fallbacks = 0
        self._ranked: list[int] = []

    def prompt(self, view: GuesserView, clue: Clue) -> str:
        board = ", ".join(view.words[i] for i in view.unrevealed)
        return GUESSER_PROMPT.format(board=board, clue=clue.word, number=clue.number)

    def next_guess(self, view: GuesserView, clue: Clue,
                   guesses_this_turn: Sequence[GuessRecord]) -> int | None:
        if len(guesses_this_turn) >= clue.number:
            return None
        if not guesses_this_turn:  # new turn: ask the LM once
            index = {view.words[i].lower(): i for i in view.unrevealed}
            reply = get_generator(self.model)(self.prompt(view, clue))
            self._ranked = [index[w] for w in parse_word_list(reply) if w in index]
            if not self._ranked:
                self.n_fallbacks += 1
                return self.fallback_guess(view)
        unrevealed = set(view.unrevealed)
        for i in self._ranked:
            if i in unrevealed:
                return i
        return None
