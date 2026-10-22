"""Clue legality and the clue type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Clue:
    word: str
    number: int

    def __str__(self) -> str:
        return f"{self.word}, {self.number}"


def is_legal_clue(clue_word: str, board_words: Iterable[str]) -> bool:
    """A clue must be a single word that is not, and does not contain/sit inside, a board word.

    Only unrevealed words strictly matter in the real game, but we check against the
    whole board to keep the rule simple and conservative.
    """
    clue_word = clue_word.lower()
    if not clue_word.isalpha():
        return False
    for w in board_words:
        w = w.lower()
        if clue_word == w or clue_word in w or w in clue_word:
            return False
    return True
