"""Learned SWOW embeddings (PPMI + SVD) on a tiny hand-made association graph."""

import numpy as np
import pytest

from codenames_rsa.meaning import EmbeddingModel, train_swow_embeddings

# Two clusters (water, animals) plus a weak bridge through "fish".
STRENGTHS = {
    "sea": {"ocean": 0.5, "water": 0.3, "fish": 0.1, "wave": 0.1},
    "ocean": {"sea": 0.6, "water": 0.3, "wave": 0.1},
    "water": {"sea": 0.3, "ocean": 0.2, "wet": 0.5},
    "wave": {"ocean": 0.5, "sea": 0.4, "water": 0.1},
    "wet": {"water": 0.8, "rain": 0.2},
    "dog": {"cat": 0.5, "pet": 0.4, "bark": 0.1},
    "cat": {"dog": 0.5, "pet": 0.4, "fish": 0.1},
    "pet": {"dog": 0.5, "cat": 0.4, "fish": 0.1},
    "bark": {"dog": 0.7, "tree": 0.3},
    "fish": {"sea": 0.4, "water": 0.3, "pet": 0.2, "cat": 0.1},
}


@pytest.fixture
def swow_emb():
    return train_swow_embeddings(STRENGTHS, dim=4)


def test_returns_embedding_model_over_cues(swow_emb):
    assert isinstance(swow_emb, EmbeddingModel)
    assert set(swow_emb.words) == set(STRENGTHS)  # "rain"/"tree" were never cues
    assert swow_emb.vectors.shape == (len(STRENGTHS), 4)


def test_clusters_are_recovered(swow_emb):
    sim = swow_emb.similarity("sea", ["ocean", "wave", "dog", "bark"])
    assert min(sim[:2]) > max(sim[2:])
    assert set(swow_emb.nearest("sea", 3)) <= {"ocean", "wave", "water", "wet"}
    # "fish" may rank near "dog": both are associated with "cat" and "pet". Embeddings
    # capture shared associates (second-order similarity), not only direct links.
    assert set(swow_emb.nearest("dog", 3)) <= {"cat", "pet", "bark", "fish"}


def test_include_non_cue_responses():
    emb = train_swow_embeddings(STRENGTHS, dim=4, cues_only=False)
    assert "rain" in emb and "tree" in emb


def test_dim_is_capped_and_deterministic():
    a = train_swow_embeddings(STRENGTHS, dim=300)
    b = train_swow_embeddings(STRENGTHS, dim=300)
    assert a.vectors.shape[1] == len(STRENGTHS) - 1
    np.testing.assert_allclose(a.vectors, b.vectors)
