"""Pretrained static embeddings (GloVe, ConceptNet Numberbatch) as a meaning model."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .base import MeaningModel


class EmbeddingModel(MeaningModel):
    """Cosine similarity over a fixed vocabulary of unit-normalized vectors."""

    def __init__(self, words: Sequence[str], vectors: np.ndarray, name: str = "embedding"):
        if len(words) != vectors.shape[0]:
            raise ValueError("words and vectors must have the same length")
        self.name = name
        self.words = list(words)
        self.index = {w: i for i, w in enumerate(self.words)}
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = (vectors / np.maximum(norms, 1e-12)).astype(np.float32)

    def __contains__(self, word: str) -> bool:
        return word in self.index

    def vector(self, word: str) -> np.ndarray:
        return self.vectors[self.index[word]]

    def similarity(self, clue: str, words: Sequence[str]) -> np.ndarray:
        return self.vectors[[self.index[w] for w in words]] @ self.vector(clue)

    def similarity_matrix(self, clues: Sequence[str], words: Sequence[str]) -> np.ndarray:
        c = self.vectors[[self.index[w] for w in clues]]
        w = self.vectors[[self.index[x] for x in words]]
        return c @ w.T

    def nearest(self, word: str, k: int) -> list[str]:
        sims = self.vectors @ self.vector(word)
        sims[self.index[word]] = -np.inf
        k = min(k, len(self.words) - 1)
        if k <= 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        return [self.words[i] for i in top[np.argsort(-sims[top])]]

    @classmethod
    def from_text_file(
        cls,
        path: str | Path,
        name: str | None = None,
        max_words: int | None = None,
        vocab: Iterable[str] | None = None,
    ) -> "EmbeddingModel":
        """Load word2vec/GloVe-style text vectors (optionally .gz).

        Handles the optional "<count> <dim>" header used by Numberbatch, and strips the
        "/c/en/" prefix from multilingual Numberbatch URIs (non-English rows are skipped).
        `vocab` restricts loading to those words, which keeps memory small.
        """
        path = Path(path)
        keep = {w.lower() for w in vocab} if vocab is not None else None
        opener = gzip.open if path.suffix == ".gz" else open
        words, rows = [], []
        with opener(path, "rt", encoding="utf8") as f:
            for line_no, line in enumerate(f):
                parts = line.rstrip().split(" ")
                if line_no == 0 and len(parts) == 2:
                    continue  # header
                word = parts[0]
                if word.startswith("/c/"):
                    if not word.startswith("/c/en/"):
                        continue
                    word = word[len("/c/en/"):]
                if not word.isalpha() or (keep is not None and word not in keep):
                    continue
                words.append(word)
                rows.append(np.asarray(parts[1:], dtype=np.float32))
                if max_words is not None and len(words) >= max_words:
                    break
        return cls(words, np.stack(rows), name=name or path.stem)
