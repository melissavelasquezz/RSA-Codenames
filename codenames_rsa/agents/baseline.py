"""No-RSA agents: embedding-similarity clue-giver and the literal (L0) guesser."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..game import CardType, Clue, GuesserView, SpymasterView
from ..game.state import GuessRecord
from ..meaning import MeaningModel
from ..rsa import CandidateGenerator
from .base import Guesser, Spymaster


class SimilaritySpymaster(Spymaster):
    """Classic baseline: pick the clue that covers the most team words while keeping each
    of them more similar to the clue than any non-team word (by `margin`).
    Ties are broken by the weakest covered team word's similarity.
    """

    name = "similarity"

    def __init__(self, meaning: MeaningModel, candidates: CandidateGenerator,
                 margin: float = 0.05, max_number: int = 4):
        self.meaning = meaning
        self.candidates = candidates
        self.margin = margin
        self.max_number = max_number

    def give_clue(self, view: SpymasterView) -> Clue:
        known = [i for i in view.unrevealed if view.words[i] in self.meaning]
        team = [i for i in known if view.types[i] == CardType.TEAM]
        bad = [i for i in known if view.types[i] != CardType.TEAM]
        team_words = [view.words[i] for i in team]
        clue_words = self.candidates.generate(team_words, view.words)
        if not clue_words:
            raise RuntimeError("No legal candidate clues; widen the candidate generator")

        team_sims = self.meaning.similarity_matrix(clue_words, team_words)
        bad_sims = (self.meaning.similarity_matrix(clue_words, [view.words[i] for i in bad])
                    if bad else np.full((len(clue_words), 1), -np.inf))

        best, best_key = None, (-1, -np.inf)
        for c, ts, bs in zip(clue_words, team_sims, bad_sims):
            threshold = bs.max() + self.margin
            ranked = np.sort(ts)[::-1][: self.max_number]
            n = int((ranked > threshold).sum())
            key = (n, ranked[max(n, 1) - 1])
            if key > best_key:
                best, best_key = Clue(c, max(n, 1)), key
        return best


class LiteralGuesser(Guesser):
    """L0 guesser: reveal the most similar unrevealed words, up to the clue number."""

    name = "L0"

    def __init__(self, meaning: MeaningModel):
        self.meaning = meaning

    def next_guess(self, view: GuesserView, clue: Clue,
                   guesses_this_turn: Sequence[GuessRecord]) -> int | None:
        if len(guesses_this_turn) >= clue.number:
            return None
        if clue.word not in self.meaning:
            return self.fallback_guess(view)
        known = [i for i in view.unrevealed if view.words[i] in self.meaning]
        if not known:
            return self.fallback_guess(view)
        sims = self.meaning.similarity(clue.word, [view.words[i] for i in known])
        return known[int(np.argmax(sims))]
