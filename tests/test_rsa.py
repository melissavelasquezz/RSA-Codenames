import numpy as np
import pytest

from codenames_rsa.agents import LiteralGuesser, PragmaticGuesser, RSASpymaster
from codenames_rsa.eval import listener_divergence, speaker_divergence, summarize
from codenames_rsa.game import Clue, GameState, play_game
from codenames_rsa.rsa import (FixedVocabCandidates, PragmaticListener, PragmaticSpeaker,
                               RSAConfig, expected_turn_values, literal_listener)

CLUES = ["sea", "wet", "pet", "burn", "song", "star", "fin"]


def test_literal_listener_normalizes(toy_meaning):
    p = literal_listener(toy_meaning, "sea", ["river", "dog", "moon"], alpha=5)
    assert p.sum() == pytest.approx(1.0)
    assert p.argmax() == 0


def test_turn_value_two_words_by_hand():
    # Word 0: team (+1, continue). Word 1: assassin (-10, stop). Equal similarity.
    ev = expected_turn_values(np.zeros(2), np.array([1.0, -10.0]), np.array([1.0, 0.0]),
                              max_guesses=2, alpha=1.0)
    assert ev[1] == pytest.approx(0.5 * 1 + 0.5 * -10)
    assert ev[2] == pytest.approx(ev[1] + 0.5 * -10)  # after a hit, only the assassin is left


def test_top_k_matches_exact_when_k_covers_board():
    rng = np.random.default_rng(0)
    sims, pay = rng.normal(size=6), rng.normal(size=6)
    cont = (pay > 0).astype(float)
    exact = expected_turn_values(sims, pay, cont, 3, alpha=2.0, top_k=None)
    full = expected_turn_values(sims, pay, cont, 3, alpha=2.0, top_k=6)
    approx = expected_turn_values(sims, pay, cont, 3, alpha=2.0, top_k=2)
    np.testing.assert_allclose(exact, full)
    assert not np.allclose(exact, approx)


def test_speaker_avoids_assassin_adjacent_clue(toy_meaning, toy_board):
    speaker = PragmaticSpeaker(toy_meaning, FixedVocabCandidates(CLUES),
                               RSAConfig(alpha=10, top_k=None))
    dist = speaker.distribution(GameState(toy_board).spymaster_view())
    best = dist.best()
    assert best.word == "wet"  # "sea"/"fin" lean toward the shark (assassin)
    assert best.number >= 2


def test_pragmatic_listener_prefers_water_words(toy_meaning, toy_board):
    listener = PragmaticListener(toy_meaning, FixedVocabCandidates(CLUES),
                                 RSAConfig(alpha=10, top_k=None))
    view = GameState(toy_board).guesser_view()
    post = listener.posterior(view, Clue("wet", 2))
    marg = post.marginals()
    top_two = sorted(marg, key=marg.get, reverse=True)[:2]
    assert {toy_board.words[i] for i in top_two} <= {"river", "ocean", "lake"}


def test_approximation_divergence_is_zero_for_identical_config(toy_meaning, toy_board):
    cfg = RSAConfig(alpha=10, top_k=None)
    cands = FixedVocabCandidates(CLUES)
    state = GameState(toy_board)
    s = PragmaticSpeaker(toy_meaning, cands, cfg)
    assert speaker_divergence(s, s, state.spymaster_view())["tv"] == pytest.approx(0)
    approx = PragmaticListener(toy_meaning, cands, RSAConfig(alpha=10, top_k=2,
                                                             l1_hypothesis_words=4))
    exact = PragmaticListener(toy_meaning, cands, RSAConfig(alpha=10, top_k=None,
                                                            l1_hypothesis_words=None))
    d = listener_divergence(exact, approx, state.guesser_view(), Clue("wet", 2))
    assert 0 <= d["tv"] <= 1


@pytest.mark.parametrize("guesser_cls", [LiteralGuesser, PragmaticGuesser])
def test_full_game_runs(toy_meaning, toy_board, guesser_cls):
    cands = FixedVocabCandidates(CLUES)
    cfg = RSAConfig(alpha=10, top_k=3)
    spymaster = RSASpymaster(toy_meaning, cands, cfg)
    guesser = (LiteralGuesser(toy_meaning) if guesser_cls is LiteralGuesser
               else PragmaticGuesser(toy_meaning, cands, cfg))
    result = play_game(spymaster, guesser, toy_board, max_clues=5)
    assert result.n_clues >= 1
    metrics = summarize([result])
    assert 0 <= metrics["assassin_rate"] <= 1
