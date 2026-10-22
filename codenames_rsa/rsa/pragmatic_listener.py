"""L1: the pragmatic listener.

L1 infers which set of words T (|T| = clue number) the speaker intended:

    P_L1(T | clue) ∝ P_S1(clue | T) · P(T)

where P_S1(clue | T) normalizes exp(beta · EV(clue | T)) over alternative clues the
speaker could have said. Under hypothesis T, words in T are treated as team cards and every
other unrevealed word has the expected payoff of a card drawn from the publicly known
remaining counts (neutral / opponent / assassin / other team cards).

Simplifications (documented so they can be revisited):
  * the speaker's number is taken as given, so only the clue word is normalized over;
  * P(T) is uniform over the pruned hypothesis set.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from ..game import CardType, Clue, GuesserView
from ..meaning import MeaningModel
from .candidates import CandidateGenerator
from .config import RSAConfig
from .literal_listener import softmax
from .turn_value import expected_turn_values


@dataclass
class ListenerPosterior:
    hypotheses: list[tuple[int, ...]]  # board indices
    probs: np.ndarray

    def marginals(self, must_include: Sequence[int] = ()) -> dict[int, float]:
        """P(word ∈ T), optionally conditioning on words already confirmed this turn."""
        keep = [h for h in range(len(self.hypotheses)) if set(must_include) <= set(self.hypotheses[h])]
        out: dict[int, float] = {}
        total = sum(self.probs[h] for h in keep)
        if total <= 0:
            return out
        for h in keep:
            for i in self.hypotheses[h]:
                out[i] = out.get(i, 0.0) + self.probs[h] / total
        return out


class PragmaticListener:
    def __init__(self, meaning: MeaningModel, candidates: CandidateGenerator, config: RSAConfig):
        self.meaning = meaning
        self.candidates = candidates
        self.config = config

    def _background(self, view: GuesserView, n: int) -> tuple[float, float]:
        """Expected payoff and continue-probability of a non-target unrevealed card."""
        counts = view.remaining_counts()
        counts[CardType.TEAM] = max(counts[CardType.TEAM] - n, 0)
        total = sum(counts.values())
        if total == 0:
            return 0.0, 0.0
        payoff = sum(c * self.config.payoffs.of(t) for t, c in counts.items()) / total
        return payoff, counts[CardType.TEAM] / total

    def posterior(self, view: GuesserView, clue: Clue) -> ListenerPosterior:
        cfg = self.config
        unrevealed = [i for i in view.unrevealed if view.words[i] in self.meaning]
        words = [view.words[i] for i in unrevealed]
        n = min(clue.number, len(words))

        # Prune hypotheses to the clue's top-m words under L0.
        sims_obs = self.meaning.similarity(clue.word, words)
        m = len(words) if cfg.l1_hypothesis_words is None else min(max(cfg.l1_hypothesis_words, n), len(words))
        hyp_pool = list(np.argsort(-sims_obs)[:m])
        hypotheses = [tuple(sorted(h)) for h in combinations(hyp_pool, n)]

        # Alternatives the speaker might have said instead (always including the observed clue).
        alts = [c for c in self.candidates.generate(words, view.words) if c != clue.word]
        if cfg.l1_max_alternatives is not None and len(alts) > cfg.l1_max_alternatives:
            alt_sims = self.meaning.similarity_matrix(alts, words)
            order = np.argsort(-alt_sims.max(axis=1))[: cfg.l1_max_alternatives]
            alts = [alts[i] for i in order]
        all_clues = [clue.word] + alts
        sims = self.meaning.similarity_matrix(all_clues, words)

        bg_payoff, bg_continue = self._background(view, n)
        log_s1 = np.empty(len(hypotheses))
        for h, targets in enumerate(hypotheses):
            payoffs = np.full(len(words), bg_payoff)
            continues = np.full(len(words), bg_continue)
            payoffs[list(targets)] = cfg.payoffs.team
            continues[list(targets)] = 1.0
            utils = np.array([
                expected_turn_values(row, payoffs, continues, n, cfg.alpha, cfg.top_k)[n]
                for row in sims
            ])
            z = cfg.beta * utils
            log_s1[h] = z[0] - (np.log(np.exp(z - z.max()).sum()) + z.max())

        probs = softmax(log_s1)
        return ListenerPosterior(
            hypotheses=[tuple(unrevealed[j] for j in h) for h in hypotheses], probs=probs
        )
