from __future__ import annotations

import codecs
import os
import re
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rich.markup import escape as escape_rich_markup

_DiagnosticStream = Literal["stdout", "stderr"]
_MAX_DIAGNOSTIC_LINES = 200
_MAX_DIAGNOSTIC_BYTES = 65_536
_MAX_DIAGNOSTIC_LINE_BYTES = 4_096
_ANSI_ESCAPE_RE = re.compile(
    r"(?:\x1b\[[0-?]*[ -/]*[@-~]"
    r"|\x9b[0-?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b[@-_])"
)
_ASSIGNMENT_SECRET_RE = re.compile(
    r"(?i)(\b(?:api[_ -]?key|token|secret|password|credential|authorization|auth)"
    r"\b\s*[:=]\s*)"
    r'(?:"[^"]*"|\'[^\']*\'|bearer\s+[^\s,;]+|[^\s,;]+)'
)
_BEARER_SECRET_RE = re.compile(r"(?i)(\bbearer\s+)[^\s,;]+")
_REDACTION = "<redacted>"


@dataclass(frozen=True, slots=True)
class AudioCppDiagnosticLine:
    """One sanitized display line captured from an owned audio.cpp child."""

    stream: _DiagnosticStream
    text: str


@dataclass(slots=True)
class _DiagnosticStreamState:
    decoder: Any
    pending: str = ""
    emitted_at_boundary: bool = False


def _new_stream_state() -> _DiagnosticStreamState:
    return _DiagnosticStreamState(
        decoder=codecs.getincrementaldecoder("utf-8")(errors="replace")
    )


def _utf8_prefix(value: str, byte_limit: int) -> str:
    if not value or byte_limit <= 0:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= byte_limit:
        return value
    return encoded[:byte_limit].decode("utf-8", errors="ignore")


def _remove_unsafe_controls(value: str) -> str:
    return "".join(
        character for character in value if unicodedata.category(character)[0] != "C"
    )


class _AudioCppDiagnosticRing:
    """Incrementally sanitize and retain a bounded child-output snapshot."""

    def __init__(self, *, home_directory: Path | None = None) -> None:
        self._home_directory = str(home_directory or Path.home())
        self._entries: deque[tuple[AudioCppDiagnosticLine, int]] = deque()
        self._retained_bytes = 0
        self._dropped_lines = 0
        self._streams = self._new_streams()

    @staticmethod
    def _new_streams() -> dict[_DiagnosticStream, _DiagnosticStreamState]:
        return {"stdout": _new_stream_state(), "stderr": _new_stream_state()}

    def feed(self, stream: _DiagnosticStream, chunk: bytes) -> None:
        """Consume one raw output chunk without retaining it.

        Args:
            stream: Child pipe that produced the chunk.
            chunk: Raw bytes read from that pipe.
        """
        state = self._streams[stream]
        decoded = state.decoder.decode(chunk, final=False)
        self._consume_decoded(stream, state, decoded)

    def finish(self, stream: _DiagnosticStream) -> None:
        """Flush a child pipe's decoder and its final unterminated line."""
        state = self._streams[stream]
        decoded = state.decoder.decode(b"", final=True)
        self._consume_decoded(stream, state, decoded)
        if state.pending:
            self._retain(stream, state.pending)
            state.pending = ""
        state.emitted_at_boundary = False
        state.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def snapshot(self) -> tuple[tuple[AudioCppDiagnosticLine, ...], int]:
        """Return immutable retained lines and the eviction count."""
        return tuple(line for line, _size in self._entries), self._dropped_lines

    def clear(self) -> None:
        """Clear all output and decoder state at a generation boundary."""
        self._entries.clear()
        self._retained_bytes = 0
        self._dropped_lines = 0
        self._streams = self._new_streams()

    def _consume_decoded(
        self,
        stream: _DiagnosticStream,
        state: _DiagnosticStreamState,
        decoded: str,
    ) -> None:
        remaining = decoded
        while remaining:
            newline = remaining.find("\n")
            if newline < 0:
                self._append_fragment(stream, state, remaining)
                return

            fragment = remaining[:newline]
            remaining = remaining[newline + 1 :]
            if fragment.endswith("\r"):
                fragment = fragment[:-1]
            self._append_fragment(stream, state, fragment)
            if state.pending or not state.emitted_at_boundary:
                self._retain(stream, state.pending)
            state.pending = ""
            state.emitted_at_boundary = False

    def _append_fragment(
        self,
        stream: _DiagnosticStream,
        state: _DiagnosticStreamState,
        fragment: str,
    ) -> None:
        remaining = fragment
        while remaining:
            capacity = _MAX_DIAGNOSTIC_LINE_BYTES - len(state.pending.encode("utf-8"))
            prefix = _utf8_prefix(remaining, capacity)
            if not prefix:
                self._retain(stream, state.pending)
                state.pending = ""
                state.emitted_at_boundary = True
                continue

            state.pending += prefix
            remaining = remaining[len(prefix) :]
            if len(state.pending.encode("utf-8")) >= _MAX_DIAGNOSTIC_LINE_BYTES:
                self._retain(stream, state.pending)
                state.pending = ""
                state.emitted_at_boundary = True
            else:
                state.emitted_at_boundary = False

    def _retain(self, stream: _DiagnosticStream, text: str) -> None:
        sanitized = self._sanitize(text)
        sanitized = _utf8_prefix(sanitized, _MAX_DIAGNOSTIC_LINE_BYTES)
        size = len(sanitized.encode("utf-8"))
        self._entries.append((AudioCppDiagnosticLine(stream, sanitized), size))
        self._retained_bytes += size

        while (
            len(self._entries) > _MAX_DIAGNOSTIC_LINES
            or self._retained_bytes > _MAX_DIAGNOSTIC_BYTES
        ):
            _line, evicted_size = self._entries.popleft()
            self._retained_bytes -= evicted_size
            self._dropped_lines += 1

    def _sanitize(self, text: str) -> str:
        value = _ANSI_ESCAPE_RE.sub("", text)
        value = _remove_unsafe_controls(value)
        value = _ASSIGNMENT_SECRET_RE.sub(rf"\1{_REDACTION}", value)
        value = _BEARER_SECRET_RE.sub(rf"\1{_REDACTION}", value)
        if self._home_directory and self._home_directory != os.sep:
            value = value.replace(self._home_directory, "~")
        return escape_rich_markup(value)
