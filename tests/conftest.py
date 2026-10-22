"""A tiny hand-built semantic space so the RSA pipeline can be tested without downloads.

Dimensions: water, animal, fire, music, space.
"""

from __future__ import annotations

import numpy as np
import pytest

from codenames_rsa.game import Board, CardType
from codenames_rsa.meaning import EmbeddingModel

TOY_VECTORS = {
    # board words
    "river": [1, 0, 0, 0, 0],
    "ocean": [1, 0.1, 0, 0, 0],
    "lake": [1, 0, 0, 0, 0.05],
    "shark": [0.8, 0.8, 0, 0, 0],
    "dog": [0, 1, 0, 0, 0],
    "horse": [0, 1, 0, 0.05, 0],
    "flame": [0, 0, 1, 0, 0],
    "piano": [0, 0, 0, 1, 0],
    "moon": [0, 0, 0, 0, 1],
    # clue-only words
    "sea": [1, 0.35, 0, 0, 0],
    "wet": [1, 0, 0.05, 0, 0],
    "pet": [0, 1, 0, 0, 0],
    "burn": [0.05, 0, 1, 0, 0],
    "song": [0, 0, 0, 1, 0],
    "star": [0, 0, 0.1, 0, 1],
    "fin": [0.6, 0.7, 0, 0, 0],
}


@pytest.fixture
def toy_meaning() -> EmbeddingModel:
    words = list(TOY_VECTORS)
    return EmbeddingModel(words, np.array([TOY_VECTORS[w] for w in words], float), name="toy")


@pytest.fixture
def toy_board() -> Board:
    """Team: the three water words. Assassin: shark (water + animal)."""
    T, N, O, A = CardType.TEAM, CardType.NEUTRAL, CardType.OPPONENT, CardType.ASSASSIN
    layout = {
        "river": T, "ocean": T, "lake": T, "shark": A, "dog": O,
        "horse": O, "flame": N, "piano": N, "moon": N,
    }
    return Board(words=tuple(layout), types=tuple(layout.values()), seed=0)
