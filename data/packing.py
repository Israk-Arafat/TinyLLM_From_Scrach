"""Pack a stream of token IDs into fixed-length chunks."""
from __future__ import annotations


class TokenPacker:
    """Accumulate token IDs and emit chunks of `chunk_size + 1` tokens
    (the extra token is needed for the labels shift)."""

    def __init__(self, chunk_size: int = 512) -> None:
        self.chunk_size = chunk_size
        self._buffer: list[int] = []

    def add(self, token_ids: list[int]) -> list[list[int]]:
        """Add token IDs and return any complete chunks."""
        self._buffer.extend(token_ids)
        return self._emit()

    def flush(self) -> list[list[int]]:
        """Return the remaining partial chunk (padded) if non-empty."""
        if len(self._buffer) > 1:
            chunk = self._buffer[: self.chunk_size + 1]
            self._buffer = []
            return [chunk]
        self._buffer = []
        return []

    def _emit(self) -> list[list[int]]:
        chunks = []
        while len(self._buffer) >= self.chunk_size + 1:
            chunks.append(self._buffer[: self.chunk_size + 1])
            self._buffer = self._buffer[self.chunk_size + 1 :]
        return chunks
