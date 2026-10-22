"""Cross-play matrix (proposal 3.2): every speaker condition against every listener condition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..conditions import MeaningRegistry, make_guesser, make_spymaster
from ..game import Board
from ..rsa import RSAConfig
from .metrics import summarize
from .simulate import run_matchup


def run_crossplay(
    speakers: Sequence[str],
    listeners: Sequence[str],
    boards: Sequence[Board],
    registry: MeaningRegistry,
    out_dir: str | Path,
    config: RSAConfig | None = None,
    speaker_meaning: str | None = None,
    listener_meaning: str | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Returns matrix[speaker][listener] -> metrics, and writes games.jsonl + matrix.json.

    Set `speaker_meaning` / `listener_meaning` (e.g. "glove" / "numberbatch") for the
    partner-mismatch condition.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    for s in speakers:
        spymaster = make_spymaster(s, registry, config, speaker_meaning)
        matrix[s] = {}
        for l in listeners:
            guesser = make_guesser(l, registry, config, listener_meaning)
            tags = {"speaker_condition": s, "listener_condition": l,
                    "speaker_meaning": speaker_meaning, "listener_meaning": listener_meaning}
            results = run_matchup(spymaster, guesser, boards, out_dir / "games.jsonl", tags)
            matrix[s][l] = summarize(results)
            print(f"[{s} -> {l}] {matrix[s][l]}")
    (out_dir / "matrix.json").write_text(json.dumps(matrix, indent=2))
    return matrix
