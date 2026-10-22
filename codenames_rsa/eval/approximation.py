"""Measure how far pruned/approximate RSA is from exact RSA on small boards (proposal 3.1)."""

from __future__ import annotations

import numpy as np

from ..game import Clue, GuesserView, SpymasterView
from ..rsa import PragmaticListener, PragmaticSpeaker


def _align(keys_a, probs_a, keys_b, probs_b):
    support = list(dict.fromkeys(list(keys_a) + list(keys_b)))
    pa = dict(zip(keys_a, probs_a))
    pb = dict(zip(keys_b, probs_b))
    return (np.array([pa.get(k, 0.0) for k in support]),
            np.array([pb.get(k, 0.0) for k in support]))


def divergences(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> dict[str, float]:
    """Total variation and KL(p || q) where p is exact, q is the approximation.

    KL is computed with `eps` smoothing because the approximation can assign zero
    probability to options it pruned away.
    """
    q_s = (q + eps) / (q + eps).sum()
    mask = p > 0
    return {
        "tv": 0.5 * float(np.abs(p - q).sum()),
        "kl": float((p[mask] * np.log(p[mask] / q_s[mask])).sum()),
        "pruned_mass": float(p[q == 0].sum()),
    }


def speaker_divergence(exact: PragmaticSpeaker, approx: PragmaticSpeaker,
                       view: SpymasterView) -> dict[str, float]:
    e, a = exact.distribution(view), approx.distribution(view)
    p, q = _align(e.clues, e.probs, a.clues, a.probs)
    return divergences(p, q)


def listener_divergence(exact: PragmaticListener, approx: PragmaticListener,
                        view: GuesserView, clue: Clue) -> dict[str, float]:
    e, a = exact.posterior(view, clue), approx.posterior(view, clue)
    p, q = _align(e.hypotheses, e.probs, a.hypotheses, a.probs)
    return divergences(p, q)
