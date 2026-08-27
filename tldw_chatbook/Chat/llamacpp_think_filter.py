"""Start-anchored, stream-aware ``<think>`` splitting for local models.

Qwen-family chat templates emit thinking at the very start of the response
(an empty ``<think>\n\n</think>`` prefix appears in no-think mode on some
generations). Only a think block that opens at the beginning of the stream
is split; a literal ``<think>`` mid-reply remains visible content. See
ADR-066 and ADR-090.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from tldw_chatbook.Chat.thinking_blocks import (
    MAX_THINKING_TEXT_BYTES,
    ThinkingEnvelopeValidationError,
)

_OPEN_TO_CLOSE = {
    "<think>": "</think>",
    "<thinking>": "</thinking>",
}
_MAX_TAG_CHARS = max(map(len, (*_OPEN_TO_CLOSE, *_OPEN_TO_CLOSE.values())))

ThinkCaptureStatus = Literal["pending", "complete", "failed"]


@dataclass(frozen=True, slots=True)
class ThinkSplitChunk:
    """One content-free-in-repr update from a thinking stream split."""

    thinking: str = field(default="", repr=False)
    content: str = field(default="", repr=False)
    status: ThinkCaptureStatus = "pending"


class StartAnchoredThinkSplitter:
    """Split a start-anchored thinking section from visible answer content.

    The undecided probe drops leading whitespace and retains only a possible
    opener prefix. A real visible first token fails the probe open and every
    later tag stays literal. A confirmed thinking block never fails open; it
    can only close, fail at EOF, or enter a terminal content-free capture
    failure.
    """

    def __init__(self) -> None:
        self._state: Literal["probing", "thinking", "visible", "failed"] = "probing"
        self._buffer = ""
        self._close_tag = ""
        self._thinking_bytes = 0
        self._strip_post_close_newlines = False
        self._terminal_status: Literal["complete", "failed"] | None = None

    def feed(self, chunk: str) -> ThinkSplitChunk:
        """Consume one stream chunk without exposing undecided tag fragments."""
        if type(chunk) is not str:
            raise TypeError("Thinking stream chunks must be strings.")
        if self._terminal_status is not None:
            raise RuntimeError("Thinking stream is already closed.")
        result = self._consume(chunk, terminal=False)
        if result is None:
            chunk = ""
            raise ThinkingEnvelopeValidationError("Invalid thinking data: text.")
        return result

    def flush(self) -> ThinkSplitChunk:
        """Settle the capture, marking an unclosed anchored block as failed."""
        if self._terminal_status is not None:
            return ThinkSplitChunk(status=self._terminal_status)
        result = self._consume("", terminal=True)
        if result is None:
            raise ThinkingEnvelopeValidationError("Invalid thinking data: text.")
        assert result.status != "pending"
        self._terminal_status = result.status
        return result

    def _consume(self, chunk: str, *, terminal: bool) -> ThinkSplitChunk | None:
        if self._state == "visible":
            return ThinkSplitChunk(
                content=self._visible_content(chunk),
                status="complete" if terminal else "pending",
            )
        if self._state == "probing":
            return self._consume_probe(chunk, terminal=terminal)
        return self._consume_thinking(chunk, terminal=terminal)

    def _consume_probe(self, chunk: str, *, terminal: bool) -> ThinkSplitChunk | None:
        if terminal:
            self._buffer = ""
            return ThinkSplitChunk(status="complete")

        for index, character in enumerate(chunk):
            if not self._buffer and character.isspace():
                continue
            self._buffer += character
            opening = next(
                (tag for tag in _OPEN_TO_CLOSE if self._buffer.startswith(tag)),
                None,
            )
            if opening is not None:
                self._state = "thinking"
                self._close_tag = _OPEN_TO_CLOSE[opening]
                self._buffer = ""
                return self._consume_thinking(chunk[index + 1 :], terminal=False)
            if not any(tag.startswith(self._buffer) for tag in _OPEN_TO_CLOSE):
                return self._fail_open_probe(chunk, index=index)
        return ThinkSplitChunk()

    def _fail_open_probe(self, chunk: str, *, index: int) -> ThinkSplitChunk:
        buffered, self._buffer = self._buffer, ""
        self._state = "visible"
        return ThinkSplitChunk(content=buffered + chunk[index + 1 :])

    def _consume_thinking(
        self, chunk: str, *, terminal: bool
    ) -> ThinkSplitChunk | None:
        if terminal:
            thinking = self._bounded_thinking((self._buffer, 0, len(self._buffer)))
            if thinking is None:
                return None
            self._buffer = ""
            return ThinkSplitChunk(thinking=thinking, status="failed")

        pending, self._buffer = self._buffer, ""
        cross_window = pending + chunk[:_MAX_TAG_CHARS]
        close_at = cross_window.find(self._close_tag)
        if close_at >= 0:
            chunk_reasoning_end = max(0, close_at - len(pending))
            thinking = self._bounded_thinking(
                (pending, 0, min(close_at, len(pending))),
                (chunk, 0, chunk_reasoning_end),
            )
            if thinking is None:
                return None
            consumed = close_at + len(self._close_tag) - len(pending)
            return self._close_thinking(thinking, chunk[consumed:])

        close_at = chunk.find(self._close_tag)
        if close_at >= 0:
            thinking = self._bounded_thinking(
                (pending, 0, len(pending)),
                (chunk, 0, close_at),
            )
            if thinking is None:
                return None
            return self._close_thinking(
                thinking, chunk[close_at + len(self._close_tag) :]
            )

        tail = pending + chunk[-(_MAX_TAG_CHARS - 1) :]
        held = self._possible_close_suffix_length(tail)
        safe_chars = len(pending) + len(chunk) - held
        pending_end = min(len(pending), safe_chars)
        chunk_end = max(0, safe_chars - len(pending))
        thinking = self._bounded_thinking(
            (pending, 0, pending_end),
            (chunk, 0, chunk_end),
        )
        if thinking is None:
            return None
        if held:
            self._buffer = (
                chunk[-held:]
                if held <= len(chunk)
                else pending[-(held - len(chunk)) :] + chunk
            )
        return ThinkSplitChunk(thinking=thinking)

    def _close_thinking(self, thinking: str, remainder: str) -> ThinkSplitChunk:
        self._state = "visible"
        self._close_tag = ""
        self._strip_post_close_newlines = True
        return ThinkSplitChunk(
            thinking=thinking,
            content=self._visible_content(remainder),
        )

    def _bounded_thinking(self, *ranges: tuple[str, int, int]) -> str | None:
        total = self._thinking_bytes
        for text, start, end in ranges:
            for index in range(start, end):
                codepoint = ord(text[index])
                if 0xD800 <= codepoint <= 0xDFFF:
                    self._terminal_capture_failure()
                    return None
                total += (
                    1
                    if codepoint <= 0x7F
                    else 2
                    if codepoint <= 0x7FF
                    else 3
                    if codepoint <= 0xFFFF
                    else 4
                )
                if total > MAX_THINKING_TEXT_BYTES:
                    self._terminal_capture_failure()
                    return None
        self._thinking_bytes = total
        return "".join(text[start:end] for text, start, end in ranges)

    def _terminal_capture_failure(self) -> None:
        self._state = "failed"
        self._buffer = ""
        self._close_tag = ""
        self._thinking_bytes = 0
        self._strip_post_close_newlines = False
        self._terminal_status = "failed"

    def _possible_close_suffix_length(self, text: str) -> int:
        maximum = min(len(text), len(self._close_tag) - 1)
        for length in range(maximum, 0, -1):
            if self._close_tag.startswith(text[-length:]):
                return length
        return 0

    def _visible_content(self, text: str) -> str:
        if not self._strip_post_close_newlines:
            return text
        content = text.lstrip("\n")
        if content:
            self._strip_post_close_newlines = False
        return content


def split_start_anchored_thinking(text: str) -> ThinkSplitChunk:
    """Split one complete response with the exact streaming semantics."""
    splitter = StartAnchoredThinkSplitter()
    update = None
    try:
        update = splitter.feed(text)
        terminal = splitter.flush()
    except ThinkingEnvelopeValidationError:
        text = ""
        update = None
        raise
    return ThinkSplitChunk(
        thinking=update.thinking + terminal.thinking,
        content=update.content + terminal.content,
        status=terminal.status,
    )


class StartAnchoredThinkFilter:
    """Compatibility wrapper that returns only the splitter's visible channel."""

    def __init__(self) -> None:
        self._splitter = StartAnchoredThinkSplitter()

    def feed(self, chunk: str) -> str:
        """Feed one stream chunk and return only visible answer content."""
        try:
            return self._splitter.feed(chunk).content
        except ThinkingEnvelopeValidationError:
            chunk = ""
            raise

    def flush(self) -> str:
        """Drop an unterminated thinking tail, preserving ADR-066 privacy."""
        return self._splitter.flush().content
