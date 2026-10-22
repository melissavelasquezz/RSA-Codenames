"""Ablation (c): a language model plays directly, with no RSA structure (control condition)."""

from __future__ import annotations

from typing import Sequence

from ..game import Clue, GuesserView, SpymasterView
from ..game.state import GuessRecord
from .base import Guesser, Spymaster


class LMSpymaster(Spymaster):
    """TODO: prompt an open-source LM (Hugging Face `transformers`) with the board and key,
    parse "<word>, <number>", and retry on illegal clues."""

    name = "lm"

    def __init__(self, model_name: str):
        self.model_name = model_name

    def give_clue(self, view: SpymasterView) -> Clue:
        raise NotImplementedError("LM spymaster is not implemented yet")


class LMGuesser(Guesser):
    """TODO: prompt an LM with the unrevealed words and the clue, parse a ranked list of
    guesses once per turn, and return them one at a time."""

    name = "lm"

    def __init__(self, model_name: str):
        self.model_name = model_name

    def next_guess(self, view: GuesserView, clue: Clue,
                   guesses_this_turn: Sequence[GuessRecord]) -> int | None:
        raise NotImplementedError("LM guesser is not implemented yet")
