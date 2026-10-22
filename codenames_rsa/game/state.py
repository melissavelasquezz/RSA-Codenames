"""Mutable game state for the single-team variant.

One team clears its cards in as few clues as possible. Opponent cards stay on the board
(and revealing one ends the turn) but the opponent never takes a turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .board import Board, CardType
from .rules import Clue


@dataclass
class GuessRecord:
    index: int
    word: str
    card_type: CardType


@dataclass
class TurnRecord:
    clue: Clue
    guesses: list[GuessRecord] = field(default_factory=list)

    @property
    def n_correct(self) -> int:
        return sum(g.card_type == CardType.TEAM for g in self.guesses)


@dataclass
class GameState:
    board: Board
    revealed: set[int] = field(default_factory=set)
    turns: list[TurnRecord] = field(default_factory=list)

    @property
    def unrevealed(self) -> list[int]:
        return [i for i in range(len(self.board)) if i not in self.revealed]

    @property
    def remaining_team(self) -> list[int]:
        return [i for i in self.board.indices_of(CardType.TEAM) if i not in self.revealed]

    @property
    def assassin_revealed(self) -> bool:
        return any(self.board.types[i] == CardType.ASSASSIN for i in self.revealed)

    @property
    def won(self) -> bool:
        return not self.remaining_team

    @property
    def lost(self) -> bool:
        return self.assassin_revealed

    @property
    def over(self) -> bool:
        return self.won or self.lost

    def reveal(self, index: int) -> CardType:
        if index in self.revealed:
            raise ValueError(f"Card {index} ({self.board.words[index]}) already revealed")
        self.revealed.add(index)
        return self.board.types[index]

    def spymaster_view(self) -> "SpymasterView":
        return SpymasterView(self)

    def guesser_view(self) -> "GuesserView":
        return GuesserView(self)


class GuesserView:
    """What a guesser may see: words, which are revealed (and their types), clue history."""

    def __init__(self, state: GameState) -> None:
        self._state = state

    @property
    def words(self) -> tuple[str, ...]:
        return self._state.board.words

    @property
    def unrevealed(self) -> list[int]:
        return self._state.unrevealed

    def revealed_type(self, index: int) -> CardType | None:
        return self._state.board.types[index] if index in self._state.revealed else None

    def remaining_counts(self) -> dict[CardType, int]:
        """Unrevealed card counts per type. Public knowledge from the board setup."""
        counts = {t: 0 for t in CardType}
        for i in self._state.unrevealed:
            counts[self._state.board.types[i]] += 1
        return counts

    @property
    def turns(self) -> list[TurnRecord]:
        return self._state.turns


class SpymasterView(GuesserView):
    """Guesser view plus the hidden key card."""

    @property
    def types(self) -> tuple[CardType, ...]:
        return self._state.board.types

    @property
    def remaining_team(self) -> list[int]:
        return self._state.remaining_team
