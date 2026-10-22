"""Task-performance metrics (proposal 3.1)."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..game import CardType, GameResult


def summarize(results: Iterable[GameResult]) -> dict[str, float]:
    """Aggregate metrics over a batch of games.

    clues_to_completion  mean clues used, over games that were won
    cards_per_clue       team cards revealed / clues given
    assassin_rate        fraction of games lost to the assassin (report this prominently!)
    opponent_rate        fraction of guesses that hit an opponent card
    neutral_rate         fraction of guesses that hit a neutral card
    """
    results = list(results)
    if not results:
        return {}
    guesses = [g for r in results for t in r.turns for g in t.guesses]
    n_guesses = max(len(guesses), 1)
    n_clues = sum(r.n_clues for r in results)
    team_hits = sum(g.card_type == CardType.TEAM for g in guesses)
    won = [r for r in results if r.won]
    return {
        "n_games": len(results),
        "win_rate": len(won) / len(results),
        "clues_to_completion": float(np.mean([r.n_clues for r in won])) if won else float("nan"),
        "cards_per_clue": team_hits / max(n_clues, 1),
        "assassin_rate": sum(r.hit_assassin for r in results) / len(results),
        "opponent_rate": sum(g.card_type == CardType.OPPONENT for g in guesses) / n_guesses,
        "neutral_rate": sum(g.card_type == CardType.NEUTRAL for g in guesses) / n_guesses,
    }
