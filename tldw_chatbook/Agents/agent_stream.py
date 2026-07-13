"""Pure streaming fence-gate for the Console agent model adapter.

Classifies a streaming turn incrementally: emit visible text as it arrives
(today's streaming UX) while never leaking fenced tool-call content, and
truncate at a disobedient mid-stream fence. No Textual, app, DB, or I/O.

``feed()`` streams only text it is certain is visible right now. A
mid-stream ``FENCE_OPEN`` occurrence becomes a *candidate* hold point the
moment it is found: streaming never advances past it. While the candidate's
tag line (the rest of its line) is not yet fully buffered, the candidate is
UNDECIDED and the gate simply waits for more chunks. If a later chunk
proves the tag line isn't clean — a look-alike such as ` ```tool_calls ` or
` ```tool_call_schema ` — the candidate is disproven: it was never a real
fence, so streaming resumes through it and the scan continues looking for
the next occurrence. Once a candidate's tag line is confirmed clean, the
gate holds for the rest of the turn — but whether the JSON body inside it
is well-formed is deliberately *not* decided here.

Finalizing (``flush_tail`` and ``result``) is the single authority for what
counts as visible: both are derived from ``split_visible_text_and_tool_call``
over the full accumulated buffer, which already scans past look-alike or
malformed fences and validates the JSON. ``flush_tail()`` returns the
authoritative visible text minus whatever has already streamed, so a held
candidate that turns out to be a look-alike or malformed JSON still flushes
its full text in the tail — nothing is ever silently dropped.

``feed()`` never streams whitespace sitting at the live edge of what it has
resolved so far (a still-undecided candidate, or simply the end of the
buffer with no candidate in sight) — a fence arriving right after it would
right-strip that whitespace away, and once streamed it can't be
un-streamed. Holding it back one chunk instead means the live-streamed
text always ends up an exact prefix of ``result()``'s visible text (see
the ``test_streamed_plus_tail_always_equals_visible`` invariant), even
under char-by-char feeding. ``flush_tail()`` still guards against a
negative-length tail (never returns fewer than zero characters) as a
defensive floor, not because an overshoot is expected in practice.
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
        self._streamed_len = 0     # chars of visible text already returned
        self._scan_from = 0        # buffer index to resume fence search from
        self._sealed = False       # leading fence: whole turn is a tool call
        self._fence_confirmed = False  # mid-stream candidate tag confirmed

    @property
    def full_text(self) -> str:
        """The complete raw accumulation — always what the loop is told."""
        return self._buf

    def _trim_trailing_ws(self, cutoff: int) -> int:
        """Pull ``cutoff`` back before any whitespace run touching it.

        Never stream whitespace sitting at the live edge of what's decided
        so far — a fence arriving right after it would rstrip it away, and
        once streamed it can't be un-streamed.
        """
        while cutoff > self._streamed_len and self._buf[cutoff - 1].isspace():
            cutoff -= 1
        return cutoff

    def feed(self, chunk: str) -> str:
        """Add a chunk and return newly-flushable visible text (may be empty)."""
        if not chunk:
            return ""
        self._buf += chunk
        if self._sealed or self._fence_confirmed:
            return ""
        verdict = stream_prefix_verdict(self._buf)
        if verdict == STREAM_TOOL_CALL:
            self._sealed = True        # leading fence: whole turn is a tool call
            return ""
        if verdict == STREAM_UNDECIDED:
            return ""                  # not enough tokens to decide — hold everything

        # STREAM_TEXT: look for a mid-stream fence candidate, resuming the
        # scan from wherever earlier look-alikes were disproven — those
        # occurrences are settled and never rechecked.
        scan_from = self._scan_from
        while True:
            idx = self._buf.find(FENCE_OPEN, scan_from)
            if idx == -1:
                # No candidate anywhere in the buffer. Stream up to a safe
                # holdback so a fence prefix split across chunks is never
                # partially shown, and never past trailing whitespace at
                # that boundary — a fence arriving in a later chunk would
                # rstrip it away, and a char-by-char feed must not commit
                # to streaming it a step early.
                self._scan_from = scan_from
                safe = max(self._streamed_len, len(self._buf) - _HOLDBACK)
                safe = self._trim_trailing_ws(safe)
                out = self._buf[self._streamed_len:safe]
                self._streamed_len = safe
                return out
            candidate = stream_prefix_verdict(self._buf[idx:])
            if candidate == STREAM_TEXT:
                # Disproven look-alike: not a hold point after all. Stream
                # through it and keep scanning for a later, real fence.
                scan_from = idx + len(FENCE_OPEN)
                continue
            if candidate == STREAM_UNDECIDED:
                # Not enough of the tag line has arrived yet to decide
                # either way. Hold here; try again on the next chunk. Also
                # hold back any trailing whitespace right before it, for
                # the same reason as above.
                self._scan_from = idx
                cutoff = self._trim_trailing_ws(idx)
                out = self._buf[self._streamed_len:cutoff]
                self._streamed_len = cutoff
                return out
            # STREAM_TOOL_CALL: tag line confirmed clean. Hold for the rest
            # of the turn — the JSON body's genuineness is settled only at
            # finalize.
            self._fence_confirmed = True
            out = self._buf[self._streamed_len:idx].rstrip()
            self._streamed_len += len(out)
            return out

    def flush_tail(self) -> str:
        """Return the authoritative remainder of visible text not yet streamed.

        Always derived from ``split_visible_text_and_tool_call`` over the
        full buffer — the single source of truth — so a held candidate
        that turns out to be a look-alike or malformed JSON still flushes
        its full text here; nothing is silently dropped.
        """
        visible, _call = split_visible_text_and_tool_call(self._buf)
        if len(visible) <= self._streamed_len:
            # A confirmed candidate's preceding text was right-stripped
            # optimistically during streaming; the authoritative visible
            # text can therefore be a hair shorter than what already
            # streamed. Nothing to add back — never return a negative tail.
            return ""
        tail = visible[self._streamed_len:]
        self._streamed_len = len(visible)
        return tail

    def result(self) -> tuple[str, ToolCall | None]:
        """Authoritative (visible_text, tool_call) over the full buffer."""
        return split_visible_text_and_tool_call(self._buf)
