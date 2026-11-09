"""Thin wrapper around an open-source chat LM (Hugging Face `transformers`).

Used by ablation (b) (`rsa.candidates.LMCandidates`) and ablation (c) (`agents.llm_agents`).
Everything that talks to an LM goes through the `TextGenerator` interface -- a function
from prompt to reply -- so tests can pass a fake generator and never download a model.
"""

from __future__ import annotations

import re
from typing import Callable

TextGenerator = Callable[[str], str]

_LOADED: dict[str, "HFChatGenerator"] = {}


class HFChatGenerator:
    """Greedy (deterministic) generation from an instruction-tuned HF model.

    `transformers` and `torch` are imported lazily, so the rest of the package works
    without the `ml` extra installed.
    """

    def __init__(self, model_name: str, max_new_tokens: int = 256, device: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        if device is not None:
            self.model.to(device)
        self.model.eval()

    def __call__(self, prompt: str) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        # return_dict=True gives {"input_ids", "attention_mask"} on every transformers version
        # (v5 returns a dict by default, older versions a bare tensor).
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)


def get_generator(model: str | TextGenerator) -> TextGenerator:
    """A model name loads (once per process) an `HFChatGenerator`; a callable is used as is."""
    if callable(model):
        return model
    if model not in _LOADED:
        _LOADED[model] = HFChatGenerator(model)
    return _LOADED[model]


def parse_word_list(text: str) -> list[str]:
    """Single-word items from an LM reply, lower-cased, in order, without duplicates.

    The reply is split on commas, semicolons and newlines; numbering ("1. ocean"), bullets
    and surrounding punctuation are stripped. An item is kept only if what remains is ONE
    word -- phrases and definitions ("Body of water where fish swim") are dropped rather
    than mined for words, which would yield junk clues like "body" or "where".
    """
    words = []
    for item in re.split(r"[,;\n]+", text):
        item = re.sub(r"^\s*(\d+[.)]|[-*\u2022])\s*", "", item)  # "1." / "2)" / "-" / bullet
        item = item.strip().strip("\"'.!?:()[]*").strip()
        if re.fullmatch(r"[A-Za-z]+", item):
            words.append(item.lower())
    return list(dict.fromkeys(words))
