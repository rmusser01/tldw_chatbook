"""Strict provider-neutral SSE framing and owned streaming primitives."""

from __future__ import annotations

import codecs
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any


_MAX_SSE_BYTES = 64 * 1024 * 1024
_MAX_SSE_LINE_CHARS = 16 * 1024 * 1024
_MAX_SSE_RECORD_CHARS = 16 * 1024 * 1024
_MAX_SSE_LINE_SEGMENTS = 65_536
_MAX_SSE_DATA_LINES = 65_536
_MAX_SSE_RECORDS = 200_000


@dataclass(frozen=True)
class SSERecord:
    """One complete Server-Sent Events record."""

    event: str | None
    data: str


class HostedSSEReadError(ValueError):
    """Raised when an owned hosted SSE response cannot be read safely."""


class SSERecordDecoder:
    """Incrementally decode strict UTF-8 Server-Sent Events records."""

    def __init__(
        self,
        *,
        max_bytes: int | None = _MAX_SSE_BYTES,
        max_line_chars: int = _MAX_SSE_LINE_CHARS,
        max_record_chars: int = _MAX_SSE_RECORD_CHARS,
        max_line_segments: int = _MAX_SSE_LINE_SEGMENTS,
        max_data_lines: int = _MAX_SSE_DATA_LINES,
        max_records: int = _MAX_SSE_RECORDS,
    ) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._line_segments: list[str] = []
        self._line_chars = 0
        self._data_lines: list[str] = []
        self._record_chars = 0
        self._event: str | None = None
        self._total_bytes = 0
        self._record_count = 0
        self._skip_leading_lf = False
        self._finished = False
        self._max_bytes = max_bytes
        self._max_line_chars = max_line_chars
        self._max_record_chars = max_record_chars
        self._max_line_segments = max_line_segments
        self._max_data_lines = max_data_lines
        self._max_records = max_records

    def feed(self, chunk: bytes) -> tuple[SSERecord, ...]:
        """Consume one response-body byte chunk.

        Args:
            chunk: Raw response bytes.

        Returns:
            Complete records dispatched by a blank line in this chunk.

        Raises:
            TypeError: If ``chunk`` is not bytes.
            UnicodeDecodeError: If the stream is not valid UTF-8.
            ValueError: If the decoder is finished or a bound is exceeded.
        """
        if self._finished:
            raise ValueError("Hosted SSE decoder is already finished.")
        if not isinstance(chunk, bytes):
            raise TypeError("Hosted SSE chunks must be bytes.")
        self._total_bytes += len(chunk)
        if self._max_bytes is not None and self._total_bytes > self._max_bytes:
            raise ValueError("Hosted SSE byte limit was exceeded.")
        return self._consume_text(self._decoder.decode(chunk, final=False))

    def finish(self) -> tuple[SSERecord, ...]:
        """Finish decoding at response-body EOF.

        Returns:
            Complete records dispatched by final decoded input.

        Raises:
            UnicodeDecodeError: If an incomplete UTF-8 sequence remains.
            ValueError: If a record/line is incomplete or already finished.
        """
        if self._finished:
            raise ValueError("Hosted SSE decoder is already finished.")
        self._finished = True
        records = self._consume_text(self._decoder.decode(b"", final=True))
        if self._line_segments or self._data_lines or self._event is not None:
            raise ValueError("Hosted SSE data record is incomplete.")
        return records

    def _consume_text(self, decoded: str) -> tuple[SSERecord, ...]:
        records: list[SSERecord] = []
        cursor = 0
        if self._skip_leading_lf:
            if decoded.startswith("\n"):
                cursor = 1
            self._skip_leading_lf = False
        segment_start = cursor
        while cursor < len(decoded):
            character = decoded[cursor]
            if character not in {"\r", "\n"}:
                cursor += 1
                continue
            self._append_line_segment(decoded[segment_start:cursor])
            records.extend(self._consume_line("".join(self._line_segments)))
            self._line_segments.clear()
            self._line_chars = 0
            cursor += 1
            if character == "\r":
                if cursor < len(decoded) and decoded[cursor] == "\n":
                    cursor += 1
                elif cursor == len(decoded):
                    self._skip_leading_lf = True
            segment_start = cursor
        self._append_line_segment(decoded[segment_start:])
        return tuple(records)

    def _append_line_segment(self, segment: str) -> None:
        if not segment:
            return
        self._line_chars += len(segment)
        if self._line_chars > self._max_line_chars:
            raise ValueError("Hosted SSE line limit was exceeded.")
        if len(self._line_segments) >= self._max_line_segments:
            raise ValueError("Hosted SSE line segment limit was exceeded.")
        self._line_segments.append(segment)

    def _consume_line(self, line: str) -> tuple[SSERecord, ...]:
        if line == "":
            if not self._data_lines:
                self._event = None
                return ()
            if self._record_count >= self._max_records:
                raise ValueError("Hosted SSE record count limit was exceeded.")
            self._record_count += 1
            record = SSERecord(event=self._event, data="\n".join(self._data_lines))
            self._data_lines.clear()
            self._record_chars = 0
            self._event = None
            return (record,)
        if line.startswith(":"):
            return ()
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            self._event = value if separator else ""
            return ()
        if field != "data":
            return ()
        if not separator:
            value = ""
        added_chars = len(value) + (1 if self._data_lines else 0)
        if self._record_chars + added_chars > self._max_record_chars:
            raise ValueError("Hosted SSE record limit was exceeded.")
        if len(self._data_lines) >= self._max_data_lines:
            raise ValueError("Hosted SSE data line limit was exceeded.")
        self._record_chars += added_chars
        self._data_lines.append(value)
        return ()


class OwnedSSEStream(Iterator[SSERecord]):
    """Own one response/session pair for its complete SSE body lifetime."""

    def __init__(self, *, response: Any, session: Any) -> None:
        self._response = response
        self._session = session
        self._chunks: Iterable[bytes] = response.iter_content(chunk_size=8192)
        self._iterator = iter(self._chunks)
        self._decoder = SSERecordDecoder()
        self._pending: deque[SSERecord] = deque()
        self._closed = False

    def __iter__(self) -> OwnedSSEStream:
        return self

    def __next__(self) -> SSERecord:
        if self._closed:
            raise StopIteration
        while not self._pending:
            try:
                chunk = next(self._iterator)
            except StopIteration:
                try:
                    self._pending.extend(self._decoder.finish())
                except Exception:
                    self.close()
                    raise HostedSSEReadError(
                        "Hosted SSE response was incomplete."
                    ) from None
                self.close()
                if not self._pending:
                    raise StopIteration
                break
            except Exception:
                self.close()
                raise HostedSSEReadError("Hosted SSE response read failed.") from None
            try:
                self._pending.extend(self._decoder.feed(chunk))
            except Exception:
                self.close()
                raise HostedSSEReadError("Hosted SSE response was malformed.") from None
        return self._pending.popleft()

    def close(self) -> None:
        """Close the response and its dedicated session exactly once."""
        if self._closed:
            return
        self._closed = True
        for resource in (self._response, self._session):
            try:
                resource.close()
            except Exception:
                pass
