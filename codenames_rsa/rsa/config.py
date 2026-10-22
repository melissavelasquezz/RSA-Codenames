from __future__ import annotations

from dataclasses import dataclass, field

from .turn_value import Payoffs


@dataclass
class RSAConfig:
    alpha: float = 10.0  # L0 rationality; scale depends on the meaning model's similarity range
    beta: float = 5.0  # S1 rationality
    max_number: int = 4  # largest clue number the speaker considers
    top_k: int | None = 5  # L0 branching per guess in the turn simulation; None = exact
    payoffs: Payoffs = field(default_factory=Payoffs)

    # L1 pruning: target-set hypotheses are drawn only from the clue's top-m L0 words,
    # and the speaker normalization uses at most this many alternative clues.
    l1_hypothesis_words: int | None = 8
    l1_max_alternatives: int | None = 100
