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

ThinkCaptureStatus = Literal["pending", "complete", "failed"]


@dataclass(frozen=True, slots=True)
class ThinkSplitChunk:
    """One content-free-in-repr update from a thinking stream split."""

    thinking: str = field(default="", repr=False)
    content: str = field(default="", repr=False)
    status: ThinkCaptureStatus = "pending"


class StartAnchoredThinkSplitter:
    """Split a start-anchored thinking section from visible answer content."""

    def __init__(self) -> None:
        self._state: Literal["probing", "thinking", "visible"] = "probing"
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
        return self._consume(chunk, terminal=False)

    def flush(self) -> ThinkSplitChunk:
        """Settle the capture, marking an unclosed anchored block as failed."""
        if self._terminal_status is not None:
            return ThinkSplitChunk(status=self._terminal_status)
        result = self._consume("", terminal=True)
        assert result.status != "pending"
        self._terminal_status = result.status
        return result

    def _consume(self, chunk: str, *, terminal: bool) -> ThinkSplitChunk:
        if self._state == "visible":
            return ThinkSplitChunk(
                content=self._visible_content(chunk),
                status="complete" if terminal else "pending",
            )

        if self._state == "probing":
            self._buffer += chunk
            stripped = self._buffer.lstrip()
            opening = next(
                (tag for tag in _OPEN_TO_CLOSE if stripped.startswith(tag)),
                None,
            )
            if opening is not None:
                self._state = "thinking"
                self._close_tag = _OPEN_TO_CLOSE[opening]
                self._buffer = stripped[len(opening) :]
            elif stripped and not any(
                tag.startswith(stripped) for tag in _OPEN_TO_CLOSE
            ):
                self._state = "visible"
                content, self._buffer = self._buffer, ""
                return ThinkSplitChunk(
                    content=content,
                    status="complete" if terminal else "pending",
                )
            else:
                status: ThinkCaptureStatus = (
                    "failed"
                    if terminal and stripped
                    else "complete"
                    if terminal
                    else "pending"
                )
                if terminal:
                    self._buffer = ""
                return ThinkSplitChunk(status=status)
        else:
            self._buffer += chunk

        close_at = self._buffer.find(self._close_tag)
        if close_at >= 0:
            thinking = self._bounded_thinking(self._buffer[:close_at])
            remainder = self._buffer[close_at + len(self._close_tag) :]
            self._buffer = ""
            self._state = "visible"
            self._strip_post_close_newlines = True
            return ThinkSplitChunk(
                thinking=thinking,
                content=self._visible_content(remainder),
                status="complete" if terminal else "pending",
            )

        if terminal:
            thinking = self._bounded_thinking(self._buffer)
            self._buffer = ""
            return ThinkSplitChunk(thinking=thinking, status="failed")

        held = self._possible_close_suffix_length(self._buffer)
        emit_through = len(self._buffer) - held
        thinking = self._bounded_thinking(self._buffer[:emit_through])
        self._buffer = self._buffer[emit_through:]
        return ThinkSplitChunk(thinking=thinking)

    def _bounded_thinking(self, text: str) -> str:
        total = self._thinking_bytes + len(text.encode("utf-8"))
        if total > MAX_THINKING_TEXT_BYTES:
            raise ThinkingEnvelopeValidationError("Invalid thinking data: text.")
        self._thinking_bytes = total
        return text

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
    update = splitter.feed(text)
    terminal = splitter.flush()
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
        return self._splitter.feed(chunk).content

    def flush(self) -> str:
        """Drop an unterminated thinking tail, preserving ADR-066 privacy."""
        return self._splitter.flush().content
