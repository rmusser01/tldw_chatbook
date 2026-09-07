"""Line-anchored, byte-exact record codec for the agent run log.

Pure module: no filesystem, no runtime imports. See the design spec
(Docs/superpowers/specs/2026-07-27-agent-programmatic-run-memory-design.md §4).

Format, one record:

    #@# 000412 run=a3f9c1 kind=primary type=tool_result tool=grep_files \
status=ok call=call_7 ts=2026-07-27T18:22:31.004Z bytes=1834
    <exactly 1834 UTF-8 bytes of content>

The header is always ONE physical line: a wrapped header would break
``^#@# `` matching and detach fields onto a continuation line. ``bytes=``
lets a parser slice content by length instead of scanning for the next
anchor, so content containing a literal ``#@#`` cannot corrupt parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

#: Never occurs naturally. ``###`` was rejected: it is a markdown H3, so
#: every heading in generated or fetched content would false-positive.
RECORD_ANCHOR = "#@#"

_ANCHOR_BYTES = RECORD_ANCHOR.encode("utf-8") + b" "
_PLACEHOLDER = "-"


def _sanitise(value: str) -> str:
    """Collapse whitespace so a value can never break the single-line header.

    Args:
        value: Raw field value (a tool name, run id, status, or call id).

    Returns:
        ``value`` with every whitespace run replaced by ``_``, or ``"-"``
        when empty.
    """
    cleaned = "_".join(str(value).split())
    return cleaned or _PLACEHOLDER


def _placeholder_to_empty(value: str) -> str:
    """Map the wire placeholder back to empty string for optional fields.

    Args:
        value: Field value from the wire (e.g., "-").

    Returns:
        Empty string if ``value`` is the placeholder, else ``value``.
    """
    return "" if value == _PLACEHOLDER else value


@dataclass(frozen=True)
class RunLogRecord:
    """One appended record: a model turn, tool call, tool result, or spawn."""

    number: int
    run_id: str
    kind: str
    type: str
    ts: str
    content: str
    tool: str = ""
    status: str = ""
    call_id: str = ""
    truncated_from: int = 0


def encode_record(record: RunLogRecord) -> bytes:
    """Serialise one record to its on-disk bytes.

    Args:
        record: The record to encode.

    Returns:
        The header line, a newline, the UTF-8 content, and a terminating
        newline. ``bytes=`` counts the content only, never the terminator.
    """
    body = record.content.encode("utf-8")
    header = (
        f"{RECORD_ANCHOR} {record.number:06d}"
        f" run={_sanitise(record.run_id)}"
        f" kind={_sanitise(record.kind)}"
        f" type={_sanitise(record.type)}"
        f" tool={_sanitise(record.tool)}"
        f" status={_sanitise(record.status)}"
        f" call={_sanitise(record.call_id)}"
        f" ts={_sanitise(record.ts)}"
        f" bytes={len(body)}"
    )
    if record.truncated_from:
        header += f" truncated={record.truncated_from}"
    return header.encode("utf-8") + b"\n" + body + b"\n"


def _parse_header(line: str) -> dict[str, str] | None:
    parts = line.split(" ")
    if len(parts) < 3 or parts[0] != RECORD_ANCHOR:
        return None
    fields: dict[str, str] = {"number": parts[1]}
    for token in parts[2:]:
        key, _, value = token.partition("=")
        if value:
            fields[key] = value
    return fields


def iter_records(data: bytes) -> Iterator[RunLogRecord]:
    """Parse every COMPLETE record in ``data``, in file order.

    A trailing record whose declared content (plus its terminating newline)
    is not fully present is skipped: the agent searches its own log while
    the writer is still appending to it, so a half-written tail is normal
    rather than corruption.

    Args:
        data: Raw bytes of one log segment.

    Yields:
        Each fully-present ``RunLogRecord``.
    """
    position = 0
    length = len(data)
    while position < length:
        if not data.startswith(_ANCHOR_BYTES, position):
            # Not at a record boundary; find the next anchor at a line start.
            nxt = data.find(b"\n" + _ANCHOR_BYTES, position)
            if nxt == -1:
                return
            position = nxt + 1
            continue
        newline = data.find(b"\n", position)
        if newline == -1:
            return
        fields = _parse_header(data[position:newline].decode("utf-8", "replace"))
        if fields is None:
            # Malformed header: resync to next anchor instead of discarding all.
            nxt = data.find(b"\n" + _ANCHOR_BYTES, position + 1)
            if nxt == -1:
                return
            position = nxt + 1
            continue
        try:
            size = int(fields.get("bytes", "0"))
            number = int(fields["number"])
        except (KeyError, ValueError):
            # Unparseable bytes or number: resync instead of losing all records.
            nxt = data.find(b"\n" + _ANCHOR_BYTES, position + 1)
            if nxt == -1:
                return
            position = nxt + 1
            continue
        # Negative size is a malformed header: resync instead of discarding all.
        if size < 0:
            nxt = data.find(b"\n" + _ANCHOR_BYTES, position + 1)
            if nxt == -1:
                return
            position = nxt + 1
            continue
        start = newline + 1
        end = start + size
        # end + 1 covers the terminating newline: only fully-terminated
        # records are yielded, so a record still being written is skipped.
        # This is a genuinely partial trailing record (normal for logs being appended).
        if end + 1 > length:
            return
        # Integrity check: the byte at the slice end MUST be the terminating newline.
        # If not, this record is torn (declared byte count overran real content);
        # resync instead of yielding stitched content.
        if data[end:end + 1] != b"\n":
            nxt = data.find(b"\n" + _ANCHOR_BYTES, position + 1)
            if nxt == -1:
                return
            position = nxt + 1
            continue
        yield RunLogRecord(
            number=number,
            run_id=fields.get("run", _PLACEHOLDER),
            kind=fields.get("kind", _PLACEHOLDER),
            type=fields.get("type", _PLACEHOLDER),
            ts=fields.get("ts", _PLACEHOLDER),
            content=data[start:end].decode("utf-8", "replace"),
            tool=_placeholder_to_empty(fields.get("tool", _PLACEHOLDER)),
            status=_placeholder_to_empty(fields.get("status", _PLACEHOLDER)),
            call_id=_placeholder_to_empty(fields.get("call", _PLACEHOLDER)),
            truncated_from=int(fields.get("truncated", "0") or 0),
        )
        position = end + 1
