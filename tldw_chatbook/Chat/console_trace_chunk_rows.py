"""Lossless physical row packing for canonical Console stream-delta events.

Packed row tags belong only to persistence. Decoding always returns ordinary
canonical event mappings, so this module does not introduce a second logical
trace-event vocabulary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Never, TypeAlias

TraceEventMapping: TypeAlias = Mapping[str, object]
TraceStorageRecord: TypeAlias = Mapping[str, object]

MIN_PACKED_RUN = 3
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_CANONICAL_KEYS = frozenset(
    {
        "event_id",
        "segment_id",
        "sequence",
        "event_type",
        "timestamp_us",
        "turn_id",
        "call_id",
        "block_id",
        "delta_kind",
        "payload",
    }
)
_ROW_KEYS = frozenset(
    {"storage_type", "segment_id", "sequence0", "timestamp0_us", "data"}
)
_DATA_KEYS = frozenset(
    {
        "event_ids",
        "turn_id",
        "call_id",
        "block_id",
        "timestamp_deltas_us",
        "payloads",
    }
)
_KIND_TO_TAG = {
    "text": "trace-text-chunks-v1",
    "reasoning": "trace-reasoning-chunks-v1",
    "tool_call": "trace-tool-call-chunks-v1",
}
_TAG_TO_KIND = {tag: kind for kind, tag in _KIND_TO_TAG.items()}


class TraceChunkRowError(ValueError):
    """Base class for claimed physical chunk-row failures."""


class TraceChunkRowFormatError(TraceChunkRowError):
    """A storage-tagged record uses an unsupported codec format."""


class TraceChunkRowCorruption(TraceChunkRowError):
    """A supported storage-tagged record is malformed or out of bounds."""


def pack_trace_event_rows(
    events: Iterable[TraceEventMapping],
) -> tuple[TraceStorageRecord, ...]:
    """Pack compatible stream-delta runs into versioned physical rows.

    Events that are not an exact supported canonical shape pass through
    verbatim. Encoding is stateless per call, so persistence batch boundaries
    can reduce compression without changing decoded history.

    Args:
        events: Canonical trace events in persistence order.

    Returns:
        Storage records in the same logical order.
    """

    packed: list[TraceStorageRecord] = []
    run: list[TraceEventMapping] = []
    run_kind: str | None = None

    def flush() -> None:
        nonlocal run, run_kind
        if run_kind is not None and len(run) >= MIN_PACKED_RUN:
            packed.append(_pack_run(run_kind, run))
        else:
            packed.extend(run)
        run = []
        run_kind = None

    for event in events:
        kind = _classify(event)
        if kind is None:
            flush()
            packed.append(event)
            continue
        if run and (kind != run_kind or not _continues(run[-1], event)):
            flush()
        run_kind = kind
        run.append(event)
    flush()
    return tuple(packed)


def decode_trace_storage_record(
    record: TraceStorageRecord,
) -> tuple[TraceEventMapping, ...]:
    """Decode one physical record into its canonical logical event sequence.

    Args:
        record: One parsed storage record.

    Returns:
        One verbatim ordinary event or all events represented by a packed row.

    Raises:
        TraceChunkRowFormatError: If a storage tag names an unsupported format.
        TraceChunkRowCorruption: If a supported row is malformed.
    """

    if "storage_type" not in record:
        return (record,)
    tag = record.get("storage_type")
    if type(tag) is not str or tag not in _TAG_TO_KIND:
        raise TraceChunkRowFormatError(
            f"unsupported trace chunk storage row: {tag!r}"
        )
    return _decode_claimed_row(record, tag)


def decode_trace_storage_records(
    records: Iterable[TraceStorageRecord],
) -> tuple[TraceEventMapping, ...]:
    """Decode storage records atomically into one canonical event sequence.

    The function returns only after every claimed packed row validates, so a
    corrupt later row cannot expose a partially decoded history to the caller.

    Args:
        records: Parsed physical records in persistence order.

    Returns:
        The complete decoded canonical event sequence.

    Raises:
        TraceChunkRowError: If any claimed packed row is unsupported or corrupt.
    """

    events: list[TraceEventMapping] = []
    for record in records:
        events.extend(decode_trace_storage_record(record))
    return tuple(events)


def _classify(event: TraceEventMapping) -> str | None:
    if set(event) != _CANONICAL_KEYS:
        return None
    if event.get("event_type") != "stream_delta":
        return None
    kind = event.get("delta_kind")
    if type(kind) is not str or kind not in _KIND_TO_TAG:
        return None
    if not _nonempty_strings(
        event,
        "event_id",
        "segment_id",
        "turn_id",
        "call_id",
        "block_id",
    ):
        return None
    if type(event.get("payload")) is not str:
        return None
    sequence = event.get("sequence")
    timestamp = event.get("timestamp_us")
    if not _is_int64(sequence) or sequence < 0 or not _is_int64(timestamp):
        return None
    return kind


def _continues(previous: TraceEventMapping, current: TraceEventMapping) -> bool:
    previous_sequence = previous["sequence"]
    current_sequence = current["sequence"]
    if current_sequence != previous_sequence + 1:
        return False
    for key in (
        "segment_id",
        "event_type",
        "delta_kind",
        "turn_id",
        "call_id",
        "block_id",
    ):
        if current[key] != previous[key]:
            return False
    return _is_int64(current["timestamp_us"] - previous["timestamp_us"])


def _pack_run(kind: str, run: list[TraceEventMapping]) -> dict[str, object]:
    first = run[0]
    return {
        "storage_type": _KIND_TO_TAG[kind],
        "segment_id": first["segment_id"],
        "sequence0": first["sequence"],
        "timestamp0_us": first["timestamp_us"],
        "data": {
            "event_ids": [event["event_id"] for event in run],
            "turn_id": first["turn_id"],
            "call_id": first["call_id"],
            "block_id": first["block_id"],
            "timestamp_deltas_us": [
                run[index]["timestamp_us"] - run[index - 1]["timestamp_us"]
                for index in range(1, len(run))
            ],
            "payloads": [event["payload"] for event in run],
        },
    }


def _decode_claimed_row(
    record: TraceStorageRecord,
    tag: str,
) -> tuple[TraceEventMapping, ...]:
    if set(record) != _ROW_KEYS:
        _malformed(tag, "envelope keys")
    segment_id = record.get("segment_id")
    if type(segment_id) is not str or not segment_id:
        _malformed(tag, "segment_id")
    sequence0 = record.get("sequence0")
    timestamp0 = record.get("timestamp0_us")
    if not _is_int64(sequence0) or sequence0 < 0:
        _malformed(tag, "sequence0")
    if not _is_int64(timestamp0):
        _malformed(tag, "timestamp0_us")
    data = record.get("data")
    if not isinstance(data, Mapping) or set(data) != _DATA_KEYS:
        _malformed(tag, "data keys")
    if not _nonempty_strings(data, "turn_id", "call_id", "block_id"):
        _malformed(tag, "turn/call/block identity")
    event_ids = data.get("event_ids")
    payloads = data.get("payloads")
    timestamp_deltas = data.get("timestamp_deltas_us")
    if (
        type(event_ids) is not list
        or len(event_ids) < MIN_PACKED_RUN
        or any(type(value) is not str or not value for value in event_ids)
    ):
        _malformed(tag, "event_ids")
    if (
        type(payloads) is not list
        or len(payloads) != len(event_ids)
        or any(type(value) is not str for value in payloads)
    ):
        _malformed(tag, "payloads")
    if (
        type(timestamp_deltas) is not list
        or len(timestamp_deltas) != len(event_ids) - 1
        or any(not _is_int64(value) for value in timestamp_deltas)
    ):
        _malformed(tag, "timestamp_deltas_us")
    if sequence0 > _INT64_MAX - (len(event_ids) - 1):
        _malformed(tag, "sequence range")

    kind = _TAG_TO_KIND[tag]
    events: list[TraceEventMapping] = []
    timestamp = timestamp0
    for index, (event_id, payload) in enumerate(zip(event_ids, payloads, strict=True)):
        if index:
            timestamp += timestamp_deltas[index - 1]
            if not _is_int64(timestamp):
                _malformed(tag, "timestamp range")
        events.append(
            {
                "event_id": event_id,
                "segment_id": segment_id,
                "sequence": sequence0 + index,
                "event_type": "stream_delta",
                "timestamp_us": timestamp,
                "turn_id": data["turn_id"],
                "call_id": data["call_id"],
                "block_id": data["block_id"],
                "delta_kind": kind,
                "payload": payload,
            }
        )
    return tuple(events)


def _nonempty_strings(record: Mapping[str, object], *keys: str) -> bool:
    return all(type(record.get(key)) is str and bool(record[key]) for key in keys)


def _is_int64(value: object) -> bool:
    return type(value) is int and _INT64_MIN <= value <= _INT64_MAX


def _malformed(tag: str, reason: str) -> Never:
    raise TraceChunkRowCorruption(f"malformed trace chunk row {tag}: {reason}")
