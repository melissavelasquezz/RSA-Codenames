# RSA-Codenames
An RSA-Based Machine Learning Study of Codenames

CS 4701 — Tanya Paul, Melissa Velasquez, Carissa Zheng.

**Question:** RSA needs a model of word meaning and a set of candidate utterances. At which
stage does machine learning help most? We build an RSA pipeline (L0 → S1 → L1) for the
single-team Codenames variant and swap one component at a time.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + tests
pip install -e ".[ml,ui]"        # LM components + Streamlit study UI
bash scripts/download_data.sh    # GloVe + Numberbatch; see data/README.md for SWOW + wordlist
pytest
```

## Layout

```
codenames_rsa/
  game/        board.py (seeded boards), rules.py (Clue, legality), state.py (views), engine.py (play_game)
  meaning/     MeaningModel interface; GloVe/Numberbatch (embeddings.py); SWOW (swow.py)
  rsa/         literal_listener.py (L0), turn_value.py (speaker utility), speaker.py (S1),
               pragmatic_listener.py (L1), candidates.py (clue-space pruning), config.py
  agents/      baseline.py (similarity spymaster, L0 guesser), rsa_agents.py (S1, L1),
               llm_agents.py (ablation c, stub)
  eval/        simulate.py (batch + JSONL), metrics.py, crossplay.py, approximation.py, qualitative.py
  conditions.py  speaker/listener condition registry + meaning-model loading
scripts/       run_crossplay.py, play_demo.py, download_data.sh
human_eval/    app.py (Streamlit study interface)
tests/         toy semantic space, game + RSA tests
```

## Model

- **L0:** `P(w | c) ∝ exp(α · sim(c, w))` over unrevealed words.
- **S1 utility:** expected value of the *whole turn*: L0 guesses sequentially without
  replacement, the turn stops on the first non-team card, and each card type has a payoff
  (`Payoffs`: team +1, neutral 0, opponent −1, assassin −10). `P(c, n | key) ∝ exp(β · EV)`.
- **L1:** infers the intended target set T (|T| = n): `P(T | c) ∝ P_S1(c | T)`, with
  non-target words given the expected payoff of the publicly known remaining card mix.
- **Tractability knobs** (`RSAConfig`): candidate generator (`NeighborCandidates`),
  `top_k` branching in the turn simulation, `l1_hypothesis_words`,
  `l1_max_alternatives`. Setting them to `None` / `FixedVocabCandidates` gives exact
  inference on small boards (`SMALL_BOARD`), and `eval/approximation.py` reports TV/KL
  against it.

## Conditions

| Speaker    | Meaning | Candidates | | Listener | |
|------------|---------|------------|-|----------|-|
| `baseline` | GloVe   | neighbors (no RSA) | | `L0` | literal |
| `rsa`      | GloVe   | neighbors  | | `L1` | pragmatic |
| `a`        | SWOW    | neighbors  | | `lm` | LM guesser |
| `b`        | GloVe   | LM         | | | |
| `a+b`      | SWOW    | LM         | | | |
| `c`        | LM plays directly | — | | | |

```bash
python scripts/run_crossplay.py --speakers baseline rsa --listeners L0 L1 --n-boards 200
python scripts/run_crossplay.py --speaker-meaning glove --listener-meaning numberbatch  # mismatch
python scripts/play_demo.py --seed 3 --speaker rsa --listener L1
streamlit run human_eval/app.py
```

## Status / TODO

- [x] Game engine, seeded boards, legality, JSONL logging, metrics
- [x] L0, S1 (expected turn value), L1, baseline agents, cross-play harness
- [x] Exact-vs-approximate divergence on small boards
- [ ] Codenames wordlist in `data/wordlists/codenames.txt`
- [ ] Learned SWOW embeddings (`meaning/swow.py:train_swow_embeddings`); currently raw association strength
- [ ] LM candidate generation (`rsa/candidates.py:LMCandidates`) and LM agents (`agents/llm_agents.py`)
- [ ] Speed: vectorize `expected_turn_values` across candidate clues (L1 is the bottleneck)
- [ ] Tune α/β per meaning model (similarity scales differ: cosine vs association strength)
- [ ] Human-eval schedules, clue-sense rating, human spymaster mode; corpus-frequency proxy
