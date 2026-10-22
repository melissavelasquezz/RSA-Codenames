"""RSA agents: S1 spymaster and L1 guesser."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..game import Board, Clue, GuesserView, SpymasterView
from ..game.state import GuessRecord
from ..meaning import MeaningModel
from ..rsa import (CandidateGenerator, ListenerPosterior, PragmaticListener,
                   PragmaticSpeaker, RSAConfig)
from .base import Guesser, Spymaster


class RSASpymaster(Spymaster):
    """Gives the S1-optimal clue (argmax; set `sample=True` to sample from S1 instead)."""

    name = "rsa"

    def __init__(self, meaning: MeaningModel, candidates: CandidateGenerator,
                 config: RSAConfig | None = None, sample: bool = False):
        self.speaker = PragmaticSpeaker(meaning, candidates, config or RSAConfig())
        self.sample = sample

    def reset(self, board: Board) -> None:
        self.rng = np.random.default_rng(board.seed)

    def give_clue(self, view: SpymasterView) -> Clue:
        dist = self.speaker.distribution(view)
        if self.sample:
            return dist.clues[self.rng.choice(len(dist.clues), p=dist.probs)]
        return dist.best()


class PragmaticGuesser(Guesser):
    """L1 guesser: guesses words by posterior probability of being an intended target,
    conditioning on targets already confirmed this turn. Stops after `clue.number` guesses
    or when the next word's conditional probability drops below `stop_threshold`.
    """

    name = "L1"

    def __init__(self, meaning: MeaningModel, candidates: CandidateGenerator,
                 config: RSAConfig | None = None, stop_threshold: float = 0.0):
        self.meaning = meaning
        self.listener = PragmaticListener(meaning, candidates, config or RSAConfig())
        self.stop_threshold = stop_threshold
        self._posterior: ListenerPosterior | None = None

    def next_guess(self, view: GuesserView, clue: Clue,
                   guesses_this_turn: Sequence[GuessRecord]) -> int | None:
        if len(guesses_this_turn) >= clue.number:
            return None
        if clue.word not in self.meaning:
            return self.fallback_guess(view)
        if not guesses_this_turn:
            self._posterior = self.listener.posterior(view, clue)

        confirmed = [g.index for g in guesses_this_turn]
        marginals = {i: p for i, p in self._posterior.marginals(confirmed).items()
                     if i not in confirmed}
        if not marginals:
            return None
        best = max(marginals, key=marginals.get)
        return best if marginals[best] >= self.stop_threshold else None
