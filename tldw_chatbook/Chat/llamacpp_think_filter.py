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
        """Feed one stream chunk and return its visible text.

        Args:
            chunk: The next assistant content chunk from the stream. May be
                empty, and may split a think tag across chunk boundaries.

        Returns:
            The portion of ``chunk`` (plus any previously buffered text)
            that is visible reply text. Empty while the stream is still
            inside a start-anchored think block or while an opening tag is
            still ambiguous. Once a non-tag start has been seen, all
            subsequent text passes through unfiltered.
        """
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
        """Signal end-of-stream and return any remaining visible tail.

        Returns:
            Always ``""``: a stream that ends while still probing or inside
            an unterminated start-anchored think block drops its tail by
            contract (ADR-066), so there is never a remaining tail to emit.
        """
        return ""
