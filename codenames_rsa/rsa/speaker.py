"""S1: the pragmatic speaker. P_S1(clue, n | key) ∝ exp(beta * EV_L0(turn | clue, n, key))."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..game import CardType, Clue, SpymasterView
from ..meaning import MeaningModel
from .candidates import CandidateGenerator
from .config import RSAConfig
from .literal_listener import softmax
from .turn_value import expected_turn_values


@dataclass
class SpeakerDistribution:
    clues: list[Clue]
    utilities: np.ndarray
    probs: np.ndarray

    def best(self) -> Clue:
        return self.clues[int(np.argmax(self.probs))]

    def top(self, k: int = 10) -> list[tuple[Clue, float, float]]:
        order = np.argsort(-self.probs)[:k]
        return [(self.clues[i], float(self.utilities[i]), float(self.probs[i])) for i in order]


class PragmaticSpeaker:
    def __init__(self, meaning: MeaningModel, candidates: CandidateGenerator, config: RSAConfig):
        self.meaning = meaning
        self.candidates = candidates
        self.config = config

    def distribution(self, view: SpymasterView) -> SpeakerDistribution:
        unrevealed = [i for i in view.unrevealed if view.words[i] in self.meaning]
        words = [view.words[i] for i in unrevealed]
        types = [view.types[i] for i in unrevealed]
        payoffs = np.array([self.config.payoffs.of(t) for t in types])
        continues = np.array([1.0 if t == CardType.TEAM else 0.0 for t in types])

        team_words = [view.words[i] for i in view.remaining_team]
        clue_words = self.candidates.generate(team_words, view.words)
        if not clue_words:
            raise RuntimeError("No legal candidate clues; widen the candidate generator")

        max_n = min(self.config.max_number, len(team_words))
        sims = self.meaning.similarity_matrix(clue_words, words)
        clues, utils = [], []
        for c, row in zip(clue_words, sims):
            ev = expected_turn_values(
                row, payoffs, continues, max_n, self.config.alpha, self.config.top_k
            )
            for n in range(1, max_n + 1):
                clues.append(Clue(c, n))
                utils.append(ev[n])

        utils_arr = np.array(utils)
        return SpeakerDistribution(clues, utils_arr, softmax(utils_arr, self.config.beta))
