"""L0: the literal listener. P_L0(w | clue) ∝ exp(alpha * sim(clue, w)) over unrevealed words."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..meaning import MeaningModel


def softmax(x: np.ndarray, temperature_scale: float = 1.0) -> np.ndarray:
    z = temperature_scale * np.asarray(x, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def literal_listener(
    meaning: MeaningModel, clue: str, words: Sequence[str], alpha: float
) -> np.ndarray:
    """Distribution over `words` (assumed to be the unrevealed cards) given `clue`."""
    return softmax(meaning.similarity(clue, words), alpha)
