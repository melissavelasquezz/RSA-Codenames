"""Expected value of a whole guessing turn under an L0 guesser.

This is the speaker's utility (proposal, Section 2). Rather than rewarding a clue for
ranking the targets highly, we simulate the turn: L0 samples a word, the turn continues
only if that word was a team card, and the next guess is drawn from L0 renormalized over
the words still on the board. Each card type has a payoff, so the trade-off between
ambition (larger numbers) and risk (hitting the assassin) falls out of the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..game import CardType
from .literal_listener import softmax


@dataclass(frozen=True)
class Payoffs:
    team: float = 1.0
    neutral: float = 0.0
    opponent: float = -1.0
    assassin: float = -10.0

    def of(self, card_type: CardType) -> float:
        return getattr(self, card_type.value)


def expected_turn_values(
    sims: np.ndarray,
    payoffs: np.ndarray,
    continue_probs: np.ndarray,
    max_guesses: int,
    alpha: float,
    top_k: int | None = None,
) -> np.ndarray:
    """Expected turn value for every guess budget 0..max_guesses in one pass.

    Args:
        sims: similarity of the clue to each unrevealed word.
        payoffs: expected payoff of revealing each word.
        continue_probs: probability the turn continues after revealing each word
            (1/0 when the card types are known; fractional under uncertainty, as in L1).
        max_guesses: largest guess budget to evaluate (the clue number).
        alpha: L0 rationality.
        top_k: if set, each step only branches on the k most likely words (renormalized).
            This is the main tractability knob; `None` is exact.

    Returns:
        Array `ev` of length max_guesses + 1 where ev[n] is the expected value of a turn
        in which the guesser takes at most n guesses.

    Note: the real rules allow number + 1 guesses; we model the guesser stopping at n.
    """
    sims = np.asarray(sims, dtype=np.float64)

    def rec(available: tuple[int, ...], budget: int) -> np.ndarray:
        ev = np.zeros(budget + 1)
        if budget == 0 or not available:
            return ev
        p = softmax(sims[list(available)], alpha)
        branch = np.arange(len(available))
        if top_k is not None and top_k < len(available):
            branch = np.argpartition(-p, top_k)[:top_k]
            p = p[branch] / p[branch].sum()
        else:
            p = p[branch]
        for j, pj in zip(branch, p):
            i = available[j]
            ev[1:] += pj * payoffs[i]
            if continue_probs[i] > 0 and budget > 1:
                sub = rec(available[:j] + available[j + 1:], budget - 1)
                ev[2:] += pj * continue_probs[i] * sub[1:]
        return ev

    return rec(tuple(range(len(sims))), max_guesses)
