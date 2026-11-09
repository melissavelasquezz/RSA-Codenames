"""Candidate clue generation: prunes the clue space the speaker (and L1) normalize over."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from ..game import is_legal_clue
from ..lm import TextGenerator, get_generator, parse_word_list
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


LM_CANDIDATE_PROMPT = """You are the spymaster in the word game Codenames.
Words on the board: {board}
Your target words: {seeds}

Suggest {n} different clues. Each clue is a SINGLE word (not a phrase or a definition) that \
relates to one or more of the target words, ideally several at once. A clue must not be, \
contain, or be part of any board word.

Example. Target words: apple, banana, cherry
Clues: fruit, orchard, pie, smoothie, tree, sweet, ripe, peel, juice, dessert

Reply with only the clues, separated by commas, on one line."""


class LMCandidates(CandidateGenerator):
    """Ablation (b): a language model proposes clues; RSA still scores them.

    The LM is prompted with the seed words and the board, its reply is parsed into single
    words, and those are filtered to legal clues the meaning model knows (RSA can only
    score those). Replies are cached per (seeds, board), since the speaker and L1 ask for
    candidates many times on the same board and generation is the slow part.

    If no proposal survives filtering, falls back to `NeighborCandidates` so the game can
    continue; `n_fallbacks` counts how often that happens so it can be reported.
    """

    def __init__(self, meaning: MeaningModel, model: str | TextGenerator,
                 n_candidates: int = 50):
        self.meaning = meaning
        self.model = model
        self.n_candidates = n_candidates
        self.n_fallbacks = 0
        self._cache: dict[tuple, list[str]] = {}
        self._fallback = NeighborCandidates(meaning)

    def prompt(self, seed_words: Sequence[str], board_words: Sequence[str]) -> str:
        return LM_CANDIDATE_PROMPT.format(
            board=", ".join(board_words), seeds=", ".join(seed_words), n=self.n_candidates)

    def generate(self, seed_words: Sequence[str], board_words: Sequence[str]) -> list[str]:
        key = (tuple(seed_words), tuple(board_words))
        if key not in self._cache:
            reply = get_generator(self.model)(self.prompt(seed_words, board_words))
            clues = [c for c in parse_word_list(reply)
                     if c in self.meaning and is_legal_clue(c, board_words)]
            if not clues:
                self.n_fallbacks += 1
                clues = self._fallback.generate(seed_words, board_words)
            self._cache[key] = clues[: self.n_candidates]
        return list(self._cache[key])
