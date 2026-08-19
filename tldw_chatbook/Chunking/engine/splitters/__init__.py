"""
Sentence splitter interface and factory.

Provides pluggable sentence splitting with a default regex-based splitter
and an optional BlingFire-backed splitter when available.
"""
from __future__ import annotations


class SentenceSplitter:
    """Interface for sentence splitting into character spans."""

    def split_to_spans(self, text: str, language: str | None = None) -> list[tuple[int, int]]:
        """Return list of (start_offset, end_offset) for sentences in text."""
        raise NotImplementedError


def get_sentence_splitter(name: str = "regex") -> SentenceSplitter:
    name = (name or "regex").strip().lower()
    if name == "blingfire":
        try:
            from .blingfire import BlingFireSentenceSplitter  # type: ignore
            return BlingFireSentenceSplitter()
        except Exception:
            # Fallback to regex if BlingFire unavailable
            from .regex import RegexSentenceSplitter
            return RegexSentenceSplitter()
    else:
        from .regex import RegexSentenceSplitter
        return RegexSentenceSplitter()
