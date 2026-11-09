"""LM spymaster and guesser (ablation c) with a scripted fake LM."""

from codenames_rsa.agents import LMGuesser, LMSpymaster
from codenames_rsa.agents.llm_agents import parse_clue
from codenames_rsa.game import Clue, GameState, play_game


class ScriptedLM:
    """Returns the scripted replies in order (repeating the last one) and records prompts."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies[min(len(self.prompts), len(self.replies)) - 1]


def test_parse_clue():
    assert parse_clue("CLUE: Water, 3") == ("water", 3)
    assert parse_clue("I would say sea 2.") == ("sea", 2)
    assert parse_clue("no idea") is None


def test_spymaster_prompt_shows_the_key(toy_board):
    lm = ScriptedLM("CLUE: wet, 3")
    clue = LMSpymaster(lm).give_clue(GameState(toy_board).spymaster_view())
    assert clue == Clue("wet", 3)
    assert "Your words (get your teammate to guess these): river, ocean, lake" in lm.prompts[0]
    assert "Assassin word (guessing it loses the game): shark" in lm.prompts[0]


def test_spymaster_clamps_number(toy_board):
    clue = LMSpymaster(ScriptedLM("CLUE: wet, 9")).give_clue(GameState(toy_board).spymaster_view())
    assert clue.number == 3  # only three team words


def test_spymaster_retries_illegal_clue_with_feedback(toy_board):
    lm = ScriptedLM("CLUE: river, 2", "CLUE: wet, 2")
    spymaster = LMSpymaster(lm)
    assert spymaster.give_clue(GameState(toy_board).spymaster_view()) == Clue("wet", 2)
    assert spymaster.n_invalid == 1
    assert "not a legal clue" in lm.prompts[1]


def test_spymaster_falls_back_after_retries(toy_board):
    spymaster = LMSpymaster(ScriptedLM("I don't know"), max_retries=1)
    clue = spymaster.give_clue(GameState(toy_board).spymaster_view())
    assert clue == Clue("thing", 1)
    assert spymaster.n_fallbacks == 1 and spymaster.n_invalid == 2


def test_guesser_follows_lm_ranking_and_asks_once_per_turn(toy_board):
    lm = ScriptedLM("Ocean, banana, river")  # "banana" is not on the board
    guesser = LMGuesser(lm)
    guesser.reset(toy_board)
    state = GameState(toy_board)
    clue = Clue("water", 2)
    first = guesser.next_guess(state.guesser_view(), clue, [])
    assert toy_board.words[first] == "ocean"
    turn = play_game(_FixedSpymaster(clue), guesser, toy_board, max_clues=1).turns[0]
    assert [g.word for g in turn.guesses] == ["ocean", "river"]
    assert len(lm.prompts) == 2  # one prompt per turn (one above, one in play_game)


def test_guesser_falls_back_when_lm_names_no_board_word(toy_board):
    guesser = LMGuesser(ScriptedLM("hmm, not sure"))
    guesser.reset(toy_board)
    view = GameState(toy_board).guesser_view()
    assert guesser.next_guess(view, Clue("water", 2), []) in view.unrevealed
    assert guesser.n_fallbacks == 1


def test_full_game_with_lm_agents(toy_board):
    spymaster = LMSpymaster(ScriptedLM("CLUE: wet, 3"))
    guesser = LMGuesser(ScriptedLM("river, ocean, lake"))
    result = play_game(spymaster, guesser, toy_board)
    assert result.won and result.n_clues == 1


class _FixedSpymaster:
    def __init__(self, clue):
        self.clue = clue

    def reset(self, board):
        pass

    def give_clue(self, view):
        return self.clue
