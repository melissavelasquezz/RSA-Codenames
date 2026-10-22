"""Proxy measures of clue style (proposal 3.4)."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..game import CardType, GameResult
from ..meaning import MeaningModel


def clues_of(results: Iterable[GameResult]) -> list[str]:
    return [t.clue.word for r in results for t in r.turns]


def type_token_ratio(clues: Sequence[str]) -> float:
    return len(set(clues)) / len(clues) if clues else float("nan")


def mean_similarity_rank(results: Iterable[GameResult], meaning: MeaningModel) -> float:
    """Average rank (1 = most similar) of each correctly guessed target among all words on
    the board at the time of the clue. Lower = more 'obvious' clues."""
    ranks = []
    for r in results:
        revealed: set[int] = set()
        for t in r.turns:
            board_idx = [i for i in range(len(r.board)) if i not in revealed
                         and r.board.words[i] in meaning]
            if t.clue.word in meaning and board_idx:
                sims = meaning.similarity(t.clue.word, [r.board.words[i] for i in board_idx])
                order = [board_idx[j] for j in np.argsort(-sims)]
                ranks += [order.index(g.index) + 1 for g in t.guesses
                          if g.card_type == CardType.TEAM and g.index in order]
            revealed.update(g.index for g in t.guesses)
    return float(np.mean(ranks)) if ranks else float("nan")


def corpus_frequency(clues: Sequence[str]) -> float:
    """TODO: mean Zipf frequency of clues (e.g. via the `wordfreq` package or SUBTLEX)."""
    raise NotImplementedError
