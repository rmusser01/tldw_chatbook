"""Pure streaming fence-gate for the Console agent model adapter.

Classifies a streaming turn incrementally: emit visible text as it arrives
(today's streaming UX) while never leaking fenced tool-call content, and
truncate at a disobedient mid-stream fence. No Textual, app, DB, or I/O.
"""
from __future__ import annotations

from .agent_models import ToolCall
from .agent_runtime import (
    FENCE_OPEN, STREAM_TEXT, STREAM_TOOL_CALL, STREAM_UNDECIDED,
    split_visible_text_and_tool_call, stream_prefix_verdict,
)

_HOLDBACK = len(FENCE_OPEN) - 1


class StreamGate:
    """Feed raw chunks; get back visible text safe to flush right now."""

    def __init__(self) -> None:
        self._buf = ""
        self._emitted = 0          # index into _buf of visible chars already flushed
        self._sealed = False       # a fence has been decided → nothing more streams

    @property
    def full_text(self) -> str:
        """The complete raw accumulation — always what the loop is told."""
        return self._buf

    def feed(self, chunk: str) -> str:
        """Add a chunk and return newly-flushable visible text (may be empty)."""
        if not chunk:
            return ""
        self._buf += chunk
        if self._sealed:
            return ""
        verdict = stream_prefix_verdict(self._buf)
        if verdict == STREAM_TOOL_CALL:
            self._sealed = True        # leading fence: whole turn is a tool call
            return ""
        if verdict == STREAM_UNDECIDED:
            return ""                  # not enough tokens to decide — hold everything
        # STREAM_TEXT: stream, but a later mid-stream fence may still truncate.
        fence = self._buf.find(FENCE_OPEN, self._emitted)
        if fence != -1:
            out = self._buf[self._emitted:fence].rstrip()
            self._emitted = fence
            self._sealed = True        # remainder is the tool call
            return out
        safe = max(self._emitted, len(self._buf) - _HOLDBACK)
        out = self._buf[self._emitted:safe]
        self._emitted = safe
        return out

    def flush_tail(self) -> str:
        """Return any held-back visible tail for a completed no-fence turn."""
        if self._sealed:
            return ""
        visible, call = split_visible_text_and_tool_call(self._buf)
        if call is not None:
            self._sealed = True
            return ""
        tail = visible[self._emitted:]
        self._emitted = len(visible)
        return tail

    def result(self) -> tuple[str, ToolCall | None]:
        """Authoritative (visible_text, tool_call) over the full buffer."""
        return split_visible_text_and_tool_call(self._buf)
