"""Tiktoken-based tokenizer wrapper."""
from __future__ import annotations

import tiktoken


class TiktokenTokenizer:
    """Thin wrapper around a tiktoken encoding with a stable interface."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size: int = self._enc.n_vocab
        self.eos_token_id: int = self._enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._enc.decode(token_ids)


def load_tokenizer(encoding_name: str = "cl100k_base") -> TiktokenTokenizer:
    return TiktokenTokenizer(encoding_name)
