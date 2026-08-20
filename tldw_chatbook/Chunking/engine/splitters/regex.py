from __future__ import annotations

import re

from . import SentenceSplitter


class RegexSentenceSplitter(SentenceSplitter):
    """Simple, dependency-free sentence splitter using regex.

    Splits on common sentence terminators . ! ? followed by whitespace or end of string.
    Returns character spans over the original text so callers can slice exactly.
    """

    # Match minimal sequences ending with one or more sentence terminators, then whitespace or end
    _pattern = re.compile(r"[^.!?]+(?:[.!?]+(?:\s+|$)|$)", re.MULTILINE)

    def split_to_spans(self, text: str, language: str | None = None) -> list[tuple[int, int]]:
        if not text:
            return []
        try:
            matches = list(self._pattern.finditer(text))
            spans = [(m.start(), m.end()) for m in matches if m and m.group(0)]
            if spans:
                return spans
        except re.error:
            pass
        # Fallback: non-empty lines
        return [(m.start(), m.end()) for m in re.finditer(r".+", text)]
