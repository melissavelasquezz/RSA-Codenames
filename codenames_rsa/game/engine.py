"""Plays a single game between a spymaster agent and a guesser agent."""

from __future__ import annotations

from dataclasses import dataclass, field

from .board import Board, CardType
from .rules import is_legal_clue
from .state import GameState, GuessRecord, TurnRecord


@dataclass
class GameResult:
    board: Board
    won: bool
    hit_assassin: bool
    n_clues: int
    turns: list[TurnRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "board": self.board.to_dict(),
            "won": self.won,
            "hit_assassin": self.hit_assassin,
            "n_clues": self.n_clues,
            "turns": [
                {
                    "clue": {"word": t.clue.word, "number": t.clue.number},
                    "guesses": [
                        {"index": g.index, "word": g.word, "type": g.card_type.value}
                        for g in t.guesses
                    ],
                }
                for t in self.turns
            ],
        }


def play_game(spymaster, guesser, board: Board, max_clues: int = 25) -> GameResult:
    """Run the single-team variant until the board is cleared, the assassin is hit, or
    `max_clues` is reached.

    Guessing follows the standard rules: up to `number + 1` guesses, and the turn ends
    on the first non-team card or when the guesser passes (returns None).
    """
    state = GameState(board)
    spymaster.reset(board)
    guesser.reset(board)

    while not state.over and len(state.turns) < max_clues:
        clue = spymaster.give_clue(state.spymaster_view())
        if not is_legal_clue(clue.word, board.words):
            raise ValueError(f"Spymaster produced illegal clue {clue!r}")
        turn = TurnRecord(clue)
        state.turns.append(turn)

        for _ in range(clue.number + 1):
            index = guesser.next_guess(state.guesser_view(), clue, turn.guesses)
            if index is None:
                break
            card_type = state.reveal(index)
            turn.guesses.append(GuessRecord(index, board.words[index], card_type))
            if card_type != CardType.TEAM or state.won:
                break

    return GameResult(
        board=board,
        won=state.won,
        hit_assassin=state.lost,
        n_clues=len(state.turns),
        turns=state.turns,
    )
