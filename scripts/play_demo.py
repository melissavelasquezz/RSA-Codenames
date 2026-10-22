"""Play one seeded game and print the clues, guesses, and S1's top alternatives.

    python scripts/play_demo.py --seed 3 --speaker rsa --listener L1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from codenames_rsa.conditions import (DATA_DIR, LISTENER_CONDITIONS, SPEAKER_CONDITIONS,
                                      MeaningRegistry, make_guesser, make_spymaster)
from codenames_rsa.game import generate_board, load_wordlist, play_game


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speaker", default="rsa", choices=list(SPEAKER_CONDITIONS))
    p.add_argument("--listener", default="L0", choices=list(LISTENER_CONDITIONS))
    p.add_argument("--wordlist", type=Path, default=DATA_DIR / "wordlists" / "codenames.txt")
    args = p.parse_args()

    registry = MeaningRegistry()
    board = generate_board(load_wordlist(args.wordlist), args.seed)
    for w, t in zip(board.words, board.types):
        print(f"  {w:<14} {t.value}")

    result = play_game(make_spymaster(args.speaker, registry),
                       make_guesser(args.listener, registry), board)
    for i, turn in enumerate(result.turns, 1):
        guesses = ", ".join(f"{g.word} ({g.card_type.value})" for g in turn.guesses)
        print(f"{i:>2}. {turn.clue}  ->  {guesses}")
    print("WON" if result.won else "LOST (assassin)" if result.hit_assassin else "OUT OF CLUES",
          f"in {result.n_clues} clues")


if __name__ == "__main__":
    main()
