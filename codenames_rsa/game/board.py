"""Board representation and seeded board generation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence


class CardType(str, Enum):
    TEAM = "team"
    NEUTRAL = "neutral"
    OPPONENT = "opponent"
    ASSASSIN = "assassin"


@dataclass(frozen=True)
class BoardConfig:
    """Card counts for a board. Defaults match the standard 5x5 game (starting team)."""

    size: int = 25
    n_team: int = 9
    n_opponent: int = 8
    n_assassin: int = 1

    @property
    def n_neutral(self) -> int:
        return self.size - self.n_team - self.n_opponent - self.n_assassin

    def __post_init__(self) -> None:
        if self.n_neutral < 0:
            raise ValueError("Card counts exceed board size")


# Small boards let us run exact RSA inference and measure approximation error (proposal 3.1).
SMALL_BOARD = BoardConfig(size=9, n_team=3, n_opponent=2, n_assassin=1)


@dataclass(frozen=True)
class Board:
    words: tuple[str, ...]
    types: tuple[CardType, ...]
    seed: int | None = None

    def __len__(self) -> int:
        return len(self.words)

    def indices_of(self, card_type: CardType) -> list[int]:
        return [i for i, t in enumerate(self.types) if t == card_type]

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "words": list(self.words),
            "types": [t.value for t in self.types],
        }


def load_wordlist(path: str | Path) -> list[str]:
    """One word per line; blank lines and #-comments ignored. Words are lower-cased."""
    words = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.append(line.lower())
    return words


def generate_board(
    wordlist: Sequence[str], seed: int, config: BoardConfig = BoardConfig()
) -> Board:
    """Deterministically sample a board from `wordlist` for the given seed."""
    rng = random.Random(seed)
    words = rng.sample(list(wordlist), config.size)
    types = (
        [CardType.TEAM] * config.n_team
        + [CardType.OPPONENT] * config.n_opponent
        + [CardType.ASSASSIN] * config.n_assassin
        + [CardType.NEUTRAL] * config.n_neutral
    )
    rng.shuffle(types)
    return Board(words=tuple(words), types=tuple(types), seed=seed)
