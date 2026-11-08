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
    strengths: dict[str, dict[str, float]],
    dim: int = 300,
    cues_only: bool = True,
    context_smoothing: float = 0.75,
    eig_power: float = 0.5,
    name: str = "swow-svd",
    seed: int = 0,
) -> EmbeddingModel:
    """Learn dense association embeddings from SWOW (ablation (a) in the proposal).

    Method: PPMI + truncated SVD (Levy, Goldberg & Dagan 2015), applied to the
    word-association graph instead of corpus co-occurrence counts.

      1. Build a word x word matrix A where A[i, j] is the strength of cue i -> response j.
         It is symmetrized (A + A^T) so "sea -> ocean" and "ocean -> sea" both count.
      2. Positive PMI: PPMI[i, j] = max(0, log P(i, j) / (P(i) P_s(j))), where P_s is the
         column marginal raised to `context_smoothing` (0.75 damps very common responses).
      3. Truncated SVD: PPMI ~ U S V^T; word vectors are U * S**eig_power.

    `cues_only` keeps only words that were themselves SWOW cues (~12k for SWOW-EN). Rare
    responses that were never cues have very sparse rows and make noisy vectors.
    The result is an `EmbeddingModel`, so it is used exactly like GloVe (cosine similarity).
    """
    from scipy import sparse
    from scipy.sparse.linalg import svds

    if cues_only:
        vocab = sorted(strengths)
    else:
        vocab = sorted(set(strengths).union(*strengths.values()))
    index = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)
    if n < 2:
        raise ValueError("need at least two words to train embeddings")

    rows, cols, vals = [], [], []
    for cue, responses in strengths.items():
        if cue not in index:
            continue
        for resp, s in responses.items():
            j = index.get(resp)
            if j is not None and j != index[cue] and s > 0:
                rows.append(index[cue])
                cols.append(j)
                vals.append(s)
    a = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64).tocsr()
    a = (a + a.T).tocoo()  # symmetrize; duplicate entries are summed

    total = a.sum()
    row_p = np.asarray(a.sum(axis=1)).ravel() / total
    col_mass = np.asarray(a.sum(axis=0)).ravel() ** context_smoothing
    col_p = col_mass / col_mass.sum()
    pmi = np.log((a.data / total) / (row_p[a.row] * col_p[a.col]))
    keep = pmi > 0
    ppmi = sparse.csr_matrix((pmi[keep], (a.row[keep], a.col[keep])), shape=(n, n))

    k = min(dim, n - 1)
    if n <= 500:  # small vocab (tests, toy boards): dense SVD is exact and fast
        u, s, _ = np.linalg.svd(ppmi.toarray())
        u, s = u[:, :k], s[:k]
    else:
        v0 = np.random.default_rng(seed).uniform(-1, 1, size=n)  # deterministic result
        u, s, _ = svds(ppmi, k=k, v0=v0)
    vectors = u * (s ** eig_power)
    return EmbeddingModel(vocab, vectors, name=name)
