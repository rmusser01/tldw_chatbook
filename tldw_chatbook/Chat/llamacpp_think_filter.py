"""Start-anchored, stream-aware ``<think>`` filtering for llama.cpp output.

Qwen-family chat templates emit thinking at the very start of the response
(an empty ``<think>\\n\\n</think>`` prefix appears in no-think mode on some
generations). Only a think block that opens at the beginning of the stream
is stripped; a literal ``<think>`` mid-reply (e.g. the user asked for an XML
example) is legitimate content and passes through. See ADR-066.
"""

from __future__ import annotations

_OPEN_TAGS = ("<think>", "<thinking>")
_CLOSE_TAGS = ("</think>", "</thinking>")


class StartAnchoredThinkFilter:
    """Stateful filter: feed() chunks in, get visible text out; flush() at end."""

    def __init__(self) -> None:
        self._inside_think = False
        self._decided_visible = False
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        if self._decided_visible:
            return chunk
        self._buffer += chunk
        while True:
            if self._inside_think:
                for tag in _CLOSE_TAGS:
                    idx = self._buffer.find(tag)
                    if idx != -1:
                        self._buffer = self._buffer[idx + len(tag):]
                        self._inside_think = False
                        self._decided_visible = True
                        return self._buffer.lstrip("\n")
                return ""
            stripped = self._buffer.lstrip()
            if not stripped:
                return ""  # whitespace-only so far; keep probing
            for tag in _OPEN_TAGS:
                if stripped.startswith(tag):
                    self._inside_think = True
                    self._buffer = stripped[len(tag):]
                    break
            if self._inside_think:
                continue  # re-run close-tag scan on the remainder
            if any(tag.startswith(stripped) for tag in _OPEN_TAGS):
                return ""  # still ambiguous: could be a split tag opener
            self._decided_visible = True
            return self._buffer

    def flush(self) -> str:
        # Stream ended while still probing or still inside an unterminated
        # think block: drop the tail (spec'd behavior).
        return ""
