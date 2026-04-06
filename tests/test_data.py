"""Tests for data pipeline."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cleaner import clean_sample
from data.packing import TokenPacker


class TestCleanSample:
    def test_returns_cleaned_text(self):
        assert clean_sample("  Hello   world  ", min_len=5) == "Hello world"

    def test_drops_short_text(self):
        assert clean_sample("Hi", min_len=50) is None

    def test_drops_non_string(self):
        assert clean_sample(None) is None  # type: ignore[arg-type]

    def test_drops_empty(self):
        assert clean_sample("", min_len=1) is None


class TestTokenPacker:
    def test_emits_full_chunks(self):
        packer = TokenPacker(chunk_size=4)
        chunks = packer.add(list(range(10)))
        assert all(len(c) == 5 for c in chunks)  # chunk_size + 1

    def test_flush_returns_remainder(self):
        packer = TokenPacker(chunk_size=4)
        packer.add([0, 1, 2])
        remainder = packer.flush()
        assert len(remainder) == 1
        assert len(remainder[0]) == 3

    def test_empty_flush(self):
        packer = TokenPacker(chunk_size=4)
        assert packer.flush() == []
