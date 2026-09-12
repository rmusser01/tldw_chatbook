"""Bounded, resumable reads of byte-framed run-log segments."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

from .run_log_format import RunLogRecord, decode_record_header

MAX_HEADER_BYTES = 16_384
_SEGMENT = re.compile(r"logs\.([0-9]+)\.txt\Z")


@dataclass(frozen=True)
class RunLogPageCursor:
    segment_index: int
    record_offset: int
    content_offset: int = 0


@dataclass(frozen=True)
class RunLogRecordSlice:
    record: RunLogRecord
    content_offset: int
    stored_content_bytes: int
    slice_complete: bool


@dataclass(frozen=True)
class RunLogPage:
    slices: tuple[RunLogRecordSlice, ...]
    start_cursor: RunLogPageCursor
    next_cursor: RunLogPageCursor | None
    scanned_bytes: int
    diagnostics: tuple[str, ...] = ()


def _next_segment(log_dir: Path, minimum: int) -> tuple[int, Path] | None:
    """Select numerically without retaining a directory-sized list."""
    chosen = None
    with os.scandir(log_dir) as entries:
        for entry in entries:
            match = _SEGMENT.fullmatch(entry.name)
            if match and entry.is_file(follow_symlinks=False):
                index = int(match[1])
                if index >= minimum and (
                    chosen is None or (index, entry.name) < (chosen[0], chosen[1].name)
                ):
                    chosen = (index, Path(entry.path))
    return chosen


def load_record_page(
    log_dir: Path,
    *,
    cursor: RunLogPageCursor | None = None,
    run_id: str | None = None,
    max_records: int = 100,
    max_content_bytes: int = 256_000,
    max_scan_bytes: int = 4_500_000,
) -> RunLogPage:
    """Read one page, preserving full stored UTF-8 content across fragments.

    Scan accounting includes every byte read (seeks do not read bodies).
    One header, a boundary/terminator check and up to four UTF-8 bytes may
    exceed the scan budget, at most MAX_HEADER_BYTES + 9 bytes per call.
    This fixed allowance guarantees progress even with tiny scan budgets.
    Content budgets below four bytes cannot hold every Unicode code point
    and are rejected. I/O errors propagate for the authority-owning caller.
    """
    return _load_page(
        log_dir, cursor, run_id, max_records, max_content_bytes, max_scan_bytes
    )


def load_record_metadata_page(
    log_dir: Path,
    *,
    cursor: RunLogPageCursor | None = None,
    run_id: str | None = None,
    max_scan_bytes: int = 4_500_000,
) -> RunLogPage:
    """Find one complete matching record without reading its content."""
    return _load_page(log_dir, cursor, run_id, 1, 0, max_scan_bytes, metadata_only=True)


def _load_page(
    log_dir,
    cursor,
    run_id,
    max_records,
    max_content_bytes,
    max_scan_bytes,
    *,
    metadata_only=False,
):
    if cursor is not None and (
        not isinstance(cursor, RunLogPageCursor)
        or any(
            type(value) is not int or value < 0
            for value in (
                cursor.segment_index,
                cursor.record_offset,
                cursor.content_offset,
            )
        )
    ):
        raise ValueError("invalid run-log cursor")
    for value, minimum in (
        (max_records, 1),
        (max_scan_bytes, 1),
        (max_content_bytes, 0 if metadata_only else 4),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError("invalid run-log page budget")
    current = cursor or RunLogPageCursor(0, 0)
    start = current
    slices = []
    diagnostics = set()
    scanned = 0
    retained = 0

    def finish(next_cursor):
        return RunLogPage(
            tuple(slices), start, next_cursor, scanned, tuple(sorted(diagnostics))
        )

    while True:
        segment = _next_segment(log_dir, current.segment_index)
        if segment is None:
            return finish(None)
        index, path = segment
        if index != current.segment_index:
            current = RunLogPageCursor(index, 0)
        with path.open("rb") as file:
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            while current.record_offset < file_size:
                if (
                    scanned >= max_scan_bytes
                    or len(slices) >= max_records
                    or (not metadata_only and retained >= max_content_bytes)
                ):
                    return finish(current)
                offset = current.record_offset
                at_line_start = True
                if offset:
                    file.seek(offset - 1)
                    at_line_start = file.read(1) == b"\n"
                    scanned += 1
                file.seek(offset)
                line = file.readline(MAX_HEADER_BYTES + 1)
                scanned += len(line)
                body_start = file.tell()
                if (
                    not at_line_start
                    or not line.endswith(b"\n")
                    or len(line) > MAX_HEADER_BYTES
                ):
                    diagnostics.add(
                        "header_too_large"
                        if len(line) > MAX_HEADER_BYTES
                        else "malformed_header"
                    )
                    current = RunLogPageCursor(index, body_start)
                    continue
                decoded = decode_record_header(line[:-1])
                if decoded is None:
                    diagnostics.add("malformed_header")
                    current = RunLogPageCursor(index, body_start)
                    continue
                record, size = decoded
                end = body_start + size
                if end >= file_size:
                    diagnostics.add("incomplete_record")
                    # A later segment can still contain complete records.
                    break
                file.seek(end)
                terminator = file.read(1)
                scanned += 1
                if terminator != b"\n":
                    diagnostics.add("torn_record")
                    current = RunLogPageCursor(index, body_start)
                    continue
                if current.content_offset > size:
                    raise ValueError("cursor exceeds stored content")
                if run_id is not None and record.run_id != run_id:
                    current = RunLogPageCursor(index, end + 1)
                    continue
                content_offset = current.content_offset
                if metadata_only:
                    content = ""
                    consumed = size - content_offset
                else:
                    allowance = min(
                        max_content_bytes - retained,
                        max(4, max_scan_bytes - scanned),
                        256_000,
                    )
                    if allowance < 4 and size - content_offset > allowance:
                        return finish(current)
                    amount = min(size - content_offset, allowance)
                    file.seek(body_start + content_offset)
                    body = file.read(amount)
                    scanned += len(body)
                    # Only a trailing partial code point is deferred; invalid
                    # stored UTF-8 is diagnosed rather than silently repaired.
                    try:
                        content = body.decode("utf-8")
                    except UnicodeDecodeError as error:
                        if error.reason == "unexpected end of data" and error.start > 0:
                            body = body[: error.start]
                            content = body.decode("utf-8")
                        else:
                            diagnostics.add("invalid_utf8")
                            current = RunLogPageCursor(index, end + 1)
                            continue
                    consumed = len(body)
                    retained += consumed
                complete = content_offset + consumed == size
                slices.append(
                    RunLogRecordSlice(
                        replace(record, content=content), content_offset, size, complete
                    )
                )
                current = (
                    RunLogPageCursor(index, end + 1)
                    if complete
                    else RunLogPageCursor(index, offset, content_offset + consumed)
                )
            current = RunLogPageCursor(index + 1, 0)


def format_record_page(page: RunLogPage) -> str:
    """Render only this page, distinguishing fragments from writer truncation."""
    blocks = []
    for part in page.slices:
        record = part.record
        header = f"record {record.number:06d} [{record.type}"
        header += f"/{record.tool}" if record.tool and record.tool != "-" else ""
        header += f"/{record.status}" if record.status and record.status != "-" else ""
        body = f"{header}]\n{record.content}"
        if part.content_offset or not part.slice_complete:
            end = part.content_offset + len(record.content.encode("utf-8"))
            body += f"\n[showing stored bytes {part.content_offset}-{end} of {part.stored_content_bytes}; {'continued' if part.slice_complete else 'continues on next page'}]"
        if record.truncated_from:
            body += f"\n[storage truncated from {record.truncated_from} bytes; {part.stored_content_bytes} bytes stored, {record.truncated_from - part.stored_content_bytes} bytes were never stored]"
        blocks.append(body)
    if page.diagnostics:
        blocks.append("[log diagnostics: " + ", ".join(page.diagnostics) + "]")
    return "\n\n".join(blocks)
