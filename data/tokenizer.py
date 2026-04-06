"""Tokenizer loading helpers."""
from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
