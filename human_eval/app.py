"""Minimal human-evaluation interface (proposal 3.3): 25 words, a clue, click to reveal.

    streamlit run human_eval/app.py

Participants act as guessers. Clues are pre-generated per (board seed, condition) so the
participant is blind to which spymaster produced them. Every click is logged to JSONL.

TODO:
  * load the participant's interleaved board/condition schedule from a JSON file
    (generated offline so clues are fixed and conditions are counterbalanced);
  * after each turn, ask for the secondary "did this clue make sense?" rating;
  * support human spymaster rounds for the human-human baseline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

from codenames_rsa.conditions import DATA_DIR
from codenames_rsa.game import CardType, generate_board, load_wordlist

LOG_PATH = Path("results/human_eval/log.jsonl")
COLORS = {CardType.TEAM: "#3b82f6", CardType.NEUTRAL: "#d6c7a1",
          CardType.OPPONENT: "#ef4444", CardType.ASSASSIN: "#111827"}


def log(event: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({"t": time.time(), **event}) + "\n")


st.set_page_config(page_title="Codenames study", layout="wide")
participant = st.sidebar.text_input("Participant ID")
seed = st.sidebar.number_input("Board seed", value=0, step=1)
clue = st.sidebar.text_input("Clue (placeholder until schedules exist)", "example")
number = st.sidebar.number_input("Number", value=2, min_value=1, max_value=9)

if not participant:
    st.info("Enter a participant ID to begin.")
    st.stop()

board = generate_board(load_wordlist(DATA_DIR / "wordlists" / "codenames.txt"), int(seed))
key = f"revealed-{participant}-{seed}"
st.session_state.setdefault(key, [])
st.session_state.setdefault("clue_shown_at", time.time())

st.header(f"Clue: {clue.upper()}, {number}")
cols = st.columns(5)
for i, word in enumerate(board.words):
    with cols[i % 5]:
        if i in st.session_state[key]:
            color = COLORS[board.types[i]]
            st.markdown(f"<div style='background:{color};color:white;padding:12px;"
                        f"text-align:center;border-radius:6px'>{word}</div>",
                        unsafe_allow_html=True)
        elif st.button(word, key=f"{key}-{i}", use_container_width=True):
            st.session_state[key].append(i)
            log({"participant": participant, "seed": int(seed), "clue": clue,
                 "number": int(number), "index": i, "word": word,
                 "type": board.types[i].value,
                 "rt": time.time() - st.session_state["clue_shown_at"]})
            st.session_state["clue_shown_at"] = time.time()
            st.rerun()
