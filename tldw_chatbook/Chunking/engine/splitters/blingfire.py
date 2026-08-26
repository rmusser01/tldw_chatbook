from __future__ import annotations

from . import SentenceSplitter


class BlingFireSentenceSplitter(SentenceSplitter):
    """Sentence splitter using BlingFire if installed.

    Falls back to a sequential search to compute spans from the newline-separated
    sentence output. If matching fails, raises to let factory fallback occur.
    """

    def __init__(self) -> None:
        # Import here to avoid hard dependency when not used
        try:
            from blingfire import text_to_sentences  # type: ignore
        except Exception as e:
            raise RuntimeError("BlingFire not available") from e
        self._b2s = text_to_sentences  # type: ignore

    def split_to_spans(self, text: str, language: str | None = None) -> list[tuple[int, int]]:
        if not text:
            return []
        sents_str = self._b2s(text)
        # BlingFire returns a string with sentences separated by newlines
        sentences = [s for s in sents_str.splitlines() if s]
        spans: list[tuple[int, int]] = []
        cursor = 0
        for s in sentences:
            idx = text.find(s, cursor)
            if idx == -1:
                # If we can't locate, fail to let the caller fallback
                raise RuntimeError("Failed to map BlingFire sentence to original text span")
            start = idx
            end = idx + len(s)
            spans.append((start, end))
            cursor = end
        return spans
