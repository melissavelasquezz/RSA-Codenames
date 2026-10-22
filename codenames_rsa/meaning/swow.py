"""Meaning models built from Small World of Words (SWOW) human association data.

Two variants are planned:
  * `AssociationModel` -- direct (symmetrized) cue->response association strength.
  * a *learned* embedding trained on SWOW (e.g. PPMI + SVD, or random-walk similarity
    as in De Deyne et al.), returned as an `EmbeddingModel` so it drops into the same
    pipeline as GloVe/Numberbatch. See `train_swow_embeddings`.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .base import MeaningModel
from .embeddings import EmbeddingModel


def load_swow_strengths(
    path: str | Path,
    cue_col: str = "cue",
    response_col: str = "response",
    strength_col: str = "R123.Strength",
    delimiter: str = "\t",
) -> dict[str, dict[str, float]]:
    """Read a SWOW strength file into {cue: {response: strength}}.

    Defaults match `strength.SWOW-EN.R123.csv` from smallworldofwords.org; adjust the
    column names if you use a different release.
    """
    strengths: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path, newline="", encoding="utf8") as f:
        for row in csv.DictReader(f, delimiter=delimiter):
            cue = row[cue_col].strip().lower()
            resp = row[response_col].strip().lower()
            if cue.isalpha() and resp.isalpha():
                strengths[cue][resp] = float(row[strength_col])
    return dict(strengths)


class AssociationModel(MeaningModel):
    """similarity(c, w) = max(P(w | cue=c), P(c | cue=w)) from SWOW."""

    def __init__(self, strengths: dict[str, dict[str, float]], name: str = "swow-assoc"):
        self.name = name
        self.strengths = strengths
        self.vocab = set(strengths)
        for responses in strengths.values():
            self.vocab.update(responses)

    def __contains__(self, word: str) -> bool:
        return word in self.vocab

    def _assoc(self, a: str, b: str) -> float:
        return max(self.strengths.get(a, {}).get(b, 0.0), self.strengths.get(b, {}).get(a, 0.0))

    def similarity(self, clue: str, words: Sequence[str]) -> np.ndarray:
        return np.array([self._assoc(clue, w) for w in words], dtype=np.float32)

    def nearest(self, word: str, k: int) -> list[str]:
        scores: dict[str, float] = dict(self.strengths.get(word, {}))
        for cue, responses in self.strengths.items():
            if word in responses:
                scores[cue] = max(scores.get(cue, 0.0), responses[word])
        scores.pop(word, None)
        return sorted(scores, key=scores.get, reverse=True)[:k]


def train_swow_embeddings(
    strengths: dict[str, dict[str, float]], dim: int = 300
) -> EmbeddingModel:
    """Learn dense association embeddings from SWOW (ablation (a) in the proposal).

    TODO: implement. Candidate approaches:
      1. PPMI on the cue x response matrix, then truncated SVD (scipy.sparse.linalg.svds).
      2. Random-walk / spreading-activation similarity (De Deyne et al. 2019), then SVD.
      3. A small PyTorch model fit to predict responses from cues.
    """
    raise NotImplementedError("SWOW embedding training is not implemented yet")
