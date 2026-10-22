"""Batch simulation over fixed seeded boards, logging every game to JSONL."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

from ..agents import Guesser, Spymaster
from ..game import Board, BoardConfig, GameResult, generate_board, play_game


def make_boards(wordlist: Sequence[str], seeds: Sequence[int],
                config: BoardConfig = BoardConfig()) -> list[Board]:
    """The same seeds give the same boards, so every condition is evaluated on paired data."""
    return [generate_board(wordlist, s, config) for s in seeds]


def run_matchup(
    spymaster: Spymaster,
    guesser: Guesser,
    boards: Sequence[Board],
    log_path: str | Path | None = None,
    tags: dict | None = None,
    max_clues: int = 25,
) -> list[GameResult]:
    """Play every board once. If `log_path` is given, append one JSON line per game."""
    results = []
    log_file = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a")
    try:
        for board in boards:
            start = time.perf_counter()
            result = play_game(spymaster, guesser, board, max_clues=max_clues)
            results.append(result)
            if log_file:
                record = {
                    **(tags or {}),
                    "spymaster": spymaster.name,
                    "guesser": guesser.name,
                    "seconds": round(time.perf_counter() - start, 3),
                    **result.to_dict(),
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
    finally:
        if log_file:
            log_file.close()
    return results
