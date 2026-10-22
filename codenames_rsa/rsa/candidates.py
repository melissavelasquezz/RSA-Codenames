"""Candidate clue generation: prunes the clue space the speaker (and L1) normalize over."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from ..game import is_legal_clue
from ..meaning import MeaningModel


class CandidateGenerator(ABC):
    @abstractmethod
    def generate(self, seed_words: Sequence[str], board_words: Sequence[str]) -> list[str]:
        """Legal candidate clues related to `seed_words`.

        The spymaster seeds with its remaining team words; L1 (which cannot see the key)
        seeds with every unrevealed word.
        """


class NeighborCandidates(CandidateGenerator):
    """Union of each seed word's k nearest neighbors in a meaning model."""

    def __init__(self, meaning: MeaningModel, k_per_word: int = 50):
        self.meaning = meaning
        self.k_per_word = k_per_word

    def generate(self, seed_words: Sequence[str], board_words: Sequence[str]) -> list[str]:
        seen: dict[str, None] = {}
        for w in seed_words:
            if w in self.meaning:
                for c in self.meaning.nearest(w, self.k_per_word):
                    seen.setdefault(c, None)
        return [c for c in seen if is_legal_clue(c, board_words)]


class FixedVocabCandidates(CandidateGenerator):
    """Every legal word from a fixed vocabulary. Used for exact inference on small boards."""

    def __init__(self, vocab: Iterable[str]):
        self.vocab = list(dict.fromkeys(vocab))

    def generate(self, seed_words: Sequence[str], board_words: Sequence[str]) -> list[str]:
        return [c for c in self.vocab if is_legal_clue(c, board_words)]


class LMCandidates(CandidateGenerator):
    """Ablation (b): a language model proposes clues; RSA still scores them.

    TODO: load an open-source LM through Hugging Face `transformers`, prompt it with the
    seed words and board, parse single-word proposals, then filter to words the meaning
    model knows (RSA can only score those) and to legal clues.
    """

    def __init__(self, meaning: MeaningModel, model_name: str, n_candidates: int = 50):
        self.meaning = meaning
        self.model_name = model_name
        self.n_candidates = n_candidates

    def generate(self, seed_words: Sequence[str], board_words: Sequence[str]) -> list[str]:
        raise NotImplementedError("LM candidate generation is not implemented yet")
