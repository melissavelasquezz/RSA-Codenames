"""Interface for word-meaning models: the lexicon that L0 is built on."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class MeaningModel(ABC):
    name: str = "meaning"

    @abstractmethod
    def __contains__(self, word: str) -> bool: ...

    @abstractmethod
    def similarity(self, clue: str, words: Sequence[str]) -> np.ndarray:
        """Relatedness of `clue` to each of `words`, shape (len(words),). Higher = closer."""

    @abstractmethod
    def nearest(self, word: str, k: int) -> list[str]:
        """The k vocabulary words most related to `word` (excluding itself)."""

    def similarity_matrix(self, clues: Sequence[str], words: Sequence[str]) -> np.ndarray:
        """Shape (len(clues), len(words)). Override for a vectorized implementation."""
        return np.stack([self.similarity(c, words) for c in clues])
