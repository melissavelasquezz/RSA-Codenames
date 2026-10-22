from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Sequence

from ..game import Board, Clue, GuesserView, SpymasterView
from ..game.state import GuessRecord


class Spymaster(ABC):
    name: str = "spymaster"

    def reset(self, board: Board) -> None:
        """Called once at the start of each game."""

    @abstractmethod
    def give_clue(self, view: SpymasterView) -> Clue: ...


class Guesser(ABC):
    name: str = "guesser"

    def reset(self, board: Board) -> None:
        self.rng = random.Random(board.seed)

    @abstractmethod
    def next_guess(
        self, view: GuesserView, clue: Clue, guesses_this_turn: Sequence[GuessRecord]
    ) -> int | None:
        """Board index to reveal next, or None to end the turn."""

    def fallback_guess(self, view: GuesserView) -> int:
        """Used when the clue is unknown to this guesser's meaning model (partner mismatch)."""
        return self.rng.choice(view.unrevealed)
