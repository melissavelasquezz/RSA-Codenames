from codenames_rsa.game import (BoardConfig, CardType, Clue, generate_board, is_legal_clue,
                                play_game)

WORDS = [f"word{chr(97 + i)}{chr(97 + j)}" for i in range(26) for j in range(2)]


def test_board_generation_is_seeded():
    a = generate_board(WORDS, seed=7)
    b = generate_board(WORDS, seed=7)
    c = generate_board(WORDS, seed=8)
    assert a == b and a != c
    assert len(a) == 25
    assert len(a.indices_of(CardType.TEAM)) == 9
    assert len(a.indices_of(CardType.ASSASSIN)) == 1


def test_board_config_validation():
    BoardConfig(size=9, n_team=3, n_opponent=2, n_assassin=1)
    try:
        BoardConfig(size=5, n_team=4, n_opponent=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_clue_legality():
    board = ["river", "fireman"]
    assert is_legal_clue("sea", board)
    assert not is_legal_clue("river", board)
    assert not is_legal_clue("fire", board)  # substring of a board word
    assert not is_legal_clue("two words", board)


class ScriptedSpymaster:
    def reset(self, board):
        pass

    def give_clue(self, view):
        return Clue("zzz", 2)


class OrderedGuesser:
    """Guesses unrevealed cards in board order."""

    def reset(self, board):
        pass

    def next_guess(self, view, clue, guesses_this_turn):
        return view.unrevealed[0]


def test_turn_ends_on_wrong_card_and_game_terminates():
    board = generate_board(WORDS, seed=1)
    result = play_game(ScriptedSpymaster(), OrderedGuesser(), board)
    assert result.won or result.hit_assassin or result.n_clues == 25
    for turn in result.turns:
        assert len(turn.guesses) <= turn.clue.number
        assert all(g.card_type == CardType.TEAM for g in turn.guesses[:-1])
