"""Run the speaker x listener cross-play matrix on seeded boards.

Examples:
    python scripts/run_crossplay.py --speakers baseline rsa --listeners L0 L1 --n-boards 100
    # partner mismatch: GloVe speaker, Numberbatch listener
    python scripts/run_crossplay.py --speaker-meaning glove --listener-meaning numberbatch
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from codenames_rsa.conditions import (DATA_DIR, LISTENER_CONDITIONS, SPEAKER_CONDITIONS,
                                      MeaningRegistry)
from codenames_rsa.eval import make_boards, run_crossplay
from codenames_rsa.game import SMALL_BOARD, BoardConfig, load_wordlist


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--speakers", nargs="+", default=["baseline", "rsa"],
                   choices=list(SPEAKER_CONDITIONS))
    p.add_argument("--listeners", nargs="+", default=["L0", "L1"],
                   choices=list(LISTENER_CONDITIONS))
    p.add_argument("--n-boards", type=int, default=100)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--small", action="store_true", help="use small 9-card boards")
    p.add_argument("--wordlist", type=Path, default=DATA_DIR / "wordlists" / "codenames.txt")
    p.add_argument("--speaker-meaning", default=None)
    p.add_argument("--listener-meaning", default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    out = args.out or Path("results") / datetime.now().strftime("crossplay-%Y%m%d-%H%M%S")
    seeds = range(args.seed_offset, args.seed_offset + args.n_boards)
    boards = make_boards(load_wordlist(args.wordlist), seeds,
                         SMALL_BOARD if args.small else BoardConfig())
    run_crossplay(args.speakers, args.listeners, boards, MeaningRegistry(), out,
                  speaker_meaning=args.speaker_meaning, listener_meaning=args.listener_meaning)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
