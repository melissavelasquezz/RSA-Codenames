"""Experimental conditions (proposal Section 2 / 3.2).

Speakers:  baseline, rsa, a (SWOW meaning), b (LM candidates), a+b, c (LM, no RSA)
Listeners: L0 (literal), L1 (pragmatic), lm

Each condition names an agent kind plus the meaning model it uses, so the partner-mismatch
experiment is just a matter of overriding the meaning key on one side.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .agents import (Guesser, LiteralGuesser, LMGuesser, LMSpymaster, PragmaticGuesser,
                     RSASpymaster, SimilaritySpymaster, Spymaster)
from .meaning import (AssociationModel, EmbeddingModel, MeaningModel, load_swow_strengths,
                      train_swow_embeddings)
from .rsa import LMCandidates, NeighborCandidates, RSAConfig

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Small enough to run on a laptop CPU; override with any HF instruction-tuned chat model.
DEFAULT_LM = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass(frozen=True)
class Condition:
    kind: str  # similarity | rsa | literal | pragmatic | lm
    meaning: str | None = "glove"
    candidates: str = "neighbors"  # neighbors | lm


SPEAKER_CONDITIONS: dict[str, Condition] = {
    "baseline": Condition("similarity"),
    "rsa": Condition("rsa"),
    "a": Condition("rsa", meaning="swow"),
    "b": Condition("rsa", candidates="lm"),
    "a+b": Condition("rsa", meaning="swow", candidates="lm"),
    "c": Condition("lm", meaning=None),
}

LISTENER_CONDITIONS: dict[str, Condition] = {
    "L0": Condition("literal"),
    "L1": Condition("pragmatic"),
    "lm": Condition("lm", meaning=None),
}


class MeaningRegistry:
    """Lazily loads and caches meaning models by key. Paths are documented in data/README.md."""

    def __init__(self, data_dir: Path = DATA_DIR, glove_max_words: int = 100_000,
                 swow_dim: int = 300):
        self.data_dir = Path(data_dir)
        self.glove_max_words = glove_max_words
        self.swow_dim = swow_dim
        self._cache: dict[str, MeaningModel] = {}

    def register(self, key: str, model: MeaningModel) -> None:
        self._cache[key] = model

    def get(self, key: str) -> MeaningModel:
        if key not in self._cache:
            self._cache[key] = self._load(key)
        return self._cache[key]

    def _load(self, key: str) -> MeaningModel:
        if key == "glove":
            return EmbeddingModel.from_text_file(
                self.data_dir / "embeddings" / "glove.6B.300d.txt",
                name="glove", max_words=self.glove_max_words)
        if key == "numberbatch":
            return EmbeddingModel.from_text_file(
                self.data_dir / "embeddings" / "numberbatch-en-19.08.txt.gz", name="numberbatch")
        if key == "swow":
            return self._load_swow_embeddings()
        if key == "swow-assoc":  # raw association strength, kept for comparison
            return AssociationModel(load_swow_strengths(self._swow_strength_path()))
        raise KeyError(f"Unknown meaning model {key!r}")

    def _swow_strength_path(self) -> Path:
        return self.data_dir / "swow" / "strength.SWOW-EN.R123.csv"

    def _load_swow_embeddings(self) -> EmbeddingModel:
        """Learned SWOW embeddings, trained once and cached next to the strength file."""
        cache = self.data_dir / "swow" / f"swow-svd-{self.swow_dim}.npz"
        if cache.exists():
            saved = np.load(cache)
            return EmbeddingModel(list(saved["words"]), saved["vectors"], name="swow")
        model = train_swow_embeddings(
            load_swow_strengths(self._swow_strength_path()), dim=self.swow_dim, name="swow")
        np.savez_compressed(cache, words=np.array(model.words), vectors=model.vectors)
        return model


def with_meaning(cond: Condition, meaning: str | None) -> Condition:
    return cond if meaning is None or cond.meaning is None else replace(cond, meaning=meaning)


def _candidates(cond: Condition, meaning: MeaningModel):
    if cond.candidates == "lm":
        return LMCandidates(meaning, DEFAULT_LM)
    return NeighborCandidates(meaning)


def make_spymaster(name: str, registry: MeaningRegistry, config: RSAConfig | None = None,
                   meaning_override: str | None = None) -> Spymaster:
    cond = with_meaning(SPEAKER_CONDITIONS[name], meaning_override)
    if cond.kind == "lm":
        return LMSpymaster(DEFAULT_LM)
    meaning = registry.get(cond.meaning)
    if cond.kind == "similarity":
        return SimilaritySpymaster(meaning, _candidates(cond, meaning))
    return RSASpymaster(meaning, _candidates(cond, meaning), config)


def make_guesser(name: str, registry: MeaningRegistry, config: RSAConfig | None = None,
                 meaning_override: str | None = None) -> Guesser:
    cond = with_meaning(LISTENER_CONDITIONS[name], meaning_override)
    if cond.kind == "lm":
        return LMGuesser(DEFAULT_LM)
    meaning = registry.get(cond.meaning)
    if cond.kind == "literal":
        return LiteralGuesser(meaning)
    return PragmaticGuesser(meaning, _candidates(cond, meaning), config)
