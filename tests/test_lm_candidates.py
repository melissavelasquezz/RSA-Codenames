"""LM clue candidates (ablation b) with a fake LM, so no model is downloaded."""

from codenames_rsa.agents import RSASpymaster
from codenames_rsa.game import GameState
from codenames_rsa.lm import parse_word_list
from codenames_rsa.rsa import LMCandidates

BOARD = ["river", "ocean", "lake", "shark", "dog", "horse", "flame", "piano", "moon"]
TEAM = ["river", "ocean", "lake"]


class FakeLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_parse_word_list_handles_numbering_and_duplicates():
    assert parse_word_list("1. Sea\n2) wet, SEA;  - star!") == ["sea", "wet", "star"]


def test_parse_word_list_drops_phrases():
    reply = "1. Waterway that flows into an ocean\n2. Tide\n3. Body of water where fish swim"
    assert parse_word_list(reply) == ["tide"]


def test_filters_illegal_unknown_and_caps(toy_meaning):
    # "lake" is a board word, "seahorse" contains "horse", "tide" is unknown to toy_meaning
    lm = FakeLM("sea, lake, seahorse, tide, wet, fin, sea")
    gen = LMCandidates(toy_meaning, lm, n_candidates=2)
    assert gen.generate(TEAM, BOARD) == ["sea", "wet"]
    assert "river, ocean, lake" in lm.prompts[0]


def test_caches_lm_calls_per_board(toy_meaning):
    lm = FakeLM("sea, wet")
    gen = LMCandidates(toy_meaning, lm)
    gen.generate(TEAM, BOARD)
    gen.generate(TEAM, BOARD)
    assert len(lm.prompts) == 1
    gen.generate(TEAM[:2], BOARD)
    assert len(lm.prompts) == 2


def test_falls_back_to_neighbors_when_nothing_usable(toy_meaning):
    gen = LMCandidates(toy_meaning, FakeLM("I am sorry, I cannot help with that."))
    clues = gen.generate(TEAM, BOARD)
    assert clues and gen.n_fallbacks == 1


def test_rsa_spymaster_with_lm_candidates(toy_meaning, toy_board):
    spymaster = RSASpymaster(toy_meaning, LMCandidates(toy_meaning, FakeLM("wet, sea, burn")))
    spymaster.reset(toy_board)
    clue = spymaster.give_clue(GameState(toy_board).spymaster_view())
    assert clue.word in {"wet", "sea", "burn"}
