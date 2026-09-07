from __future__ import annotations

from collections.abc import Mapping

import pytest

from tldw_chatbook.Chat.console_trace_chunk_rows import (
    TraceChunkRowCorruption,
    TraceChunkRowFormatError,
    decode_trace_storage_record,
    decode_trace_storage_records,
    pack_trace_event_rows,
)


def _delta(
    sequence: int,
    payload: str,
    *,
    kind: str = "text",
    timestamp_us: int | None = None,
    event_id: str | None = None,
    segment_id: str = "segment-a",
    turn_id: str = "turn-a",
    call_id: str = "call-a",
    block_id: str = "block-a",
) -> dict[str, object]:
    return {
        "event_id": event_id or f"event-{sequence}",
        "segment_id": segment_id,
        "sequence": sequence,
        "event_type": "stream_delta",
        "timestamp_us": sequence * 10 if timestamp_us is None else timestamp_us,
        "turn_id": turn_id,
        "call_id": call_id,
        "block_id": block_id,
        "delta_kind": kind,
        "payload": payload,
    }


def _round_trip(events: list[dict[str, object]]) -> tuple[Mapping[str, object], ...]:
    return decode_trace_storage_records(pack_trace_event_rows(events))


def test_text_run_packs_once_and_round_trips_unicode_and_empty_boundaries() -> None:
    events = [
        _delta(0, ""),
        _delta(1, "hé"),
        _delta(2, "🙂"),
        _delta(3, "\u0000tail"),
    ]

    rows = pack_trace_event_rows(events)

    assert len(rows) == 1
    assert rows[0]["storage_type"] == "trace-text-chunks-v1"
    assert "event_type" not in rows[0]
    assert _round_trip(events) == tuple(events)


def test_reasoning_run_preserves_positive_zero_and_negative_timestamp_gaps() -> None:
    events = [
        _delta(7, "a", kind="reasoning", timestamp_us=1_000),
        _delta(8, "b", kind="reasoning", timestamp_us=1_050),
        _delta(9, "c", kind="reasoning", timestamp_us=1_050),
        _delta(10, "d", kind="reasoning", timestamp_us=900),
    ]

    rows = pack_trace_event_rows(events)

    assert rows[0]["storage_type"] == "trace-reasoning-chunks-v1"
    assert rows[0]["data"]["timestamp_deltas_us"] == [50, 0, -150]
    assert decode_trace_storage_records(rows) == tuple(events)


def test_tool_call_runs_require_one_exact_block_identity() -> None:
    first = [
        _delta(index, payload, kind="tool_call", block_id="tool-1")
        for index, payload in enumerate(("{", '"x":', "1"))
    ]
    second = [
        _delta(index, payload, kind="tool_call", block_id="tool-2")
        for index, payload in enumerate(("[", "2", "]"), start=3)
    ]

    rows = pack_trace_event_rows([*first, *second])

    assert [row["storage_type"] for row in rows] == [
        "trace-tool-call-chunks-v1",
        "trace-tool-call-chunks-v1",
    ]
    assert decode_trace_storage_records(rows) == tuple([*first, *second])


@pytest.mark.parametrize(
    "changed",
    [
        {"delta_kind": "reasoning"},
        {"segment_id": "segment-b"},
        {"turn_id": "turn-b"},
        {"call_id": "call-b"},
        {"block_id": "block-b"},
        {"sequence": 8},
    ],
)
def test_incompatible_delta_breaks_a_run_without_reordering(
    changed: dict[str, object],
) -> None:
    events = [_delta(0, "a"), _delta(1, "b")]
    third = _delta(2, "c")
    third.update(changed)
    events.append(third)

    rows = pack_trace_event_rows(events)

    assert rows == tuple(events)
    assert decode_trace_storage_records(rows) == tuple(events)


def test_structural_extended_and_short_events_remain_verbatim_boundaries() -> None:
    structural = {
        "event_id": "boundary",
        "segment_id": "segment-a",
        "sequence": 3,
        "event_type": "call_outcome",
        "call_id": "call-a",
    }
    extended = _delta(7, "future")
    extended["future_field"] = True
    events = [
        _delta(0, "a"),
        _delta(1, "b"),
        _delta(2, "c"),
        structural,
        _delta(4, "d"),
        _delta(5, "e"),
        _delta(6, "f"),
        extended,
    ]

    rows = pack_trace_event_rows(events)

    assert len(rows) == 4
    assert rows[1] is structural
    assert rows[-1] is extended
    assert decode_trace_storage_records(rows) == tuple(events)


def test_split_persistence_batches_need_no_run_alignment() -> None:
    events = [_delta(index, str(index)) for index in range(8)]

    split_rows = [
        *pack_trace_event_rows(events[:2]),
        *pack_trace_event_rows(events[2:5]),
        *pack_trace_event_rows(events[5:]),
    ]

    assert len(split_rows) == 4
    assert decode_trace_storage_records(split_rows) == tuple(events)


def test_interrupted_stream_packs_without_a_terminal_event() -> None:
    events = [_delta(index, value) for index, value in enumerate(("one", "two", ""))]

    rows = pack_trace_event_rows(events)

    assert len(rows) == 1
    assert decode_trace_storage_records(rows) == tuple(events)


def test_existing_unencoded_trace_record_reads_unchanged() -> None:
    event = {
        "event_id": "existing",
        "segment_id": "segment-a",
        "sequence": 0,
        "event_type": "surface_append",
        "turn_id": "turn-a",
        "surface_node_id": "surface-a",
    }

    assert decode_trace_storage_record(event) == (event,)
    assert decode_trace_storage_records((event,)) == (event,)


@pytest.mark.parametrize(
    "record",
    [
        {"storage_type": "trace-text-chunks-v1"},
        {
            "storage_type": "trace-text-chunks-v1",
            "segment_id": "segment-a",
            "sequence0": 0,
            "timestamp0_us": 0,
            "data": {
                "event_ids": ["a", "b", "c"],
                "turn_id": "turn-a",
                "call_id": "call-a",
                "block_id": "block-a",
                "timestamp_deltas_us": [1],
                "payloads": ["a", "b", "c"],
            },
        },
        {
            "storage_type": "trace-tool-call-chunks-v1",
            "segment_id": "segment-a",
            "sequence0": 0,
            "timestamp0_us": 0,
            "data": {
                "event_ids": ["a", "b", ""],
                "turn_id": "turn-a",
                "call_id": "call-a",
                "block_id": "block-a",
                "timestamp_deltas_us": [1, 1],
                "payloads": ["a", "b", "c"],
            },
        },
    ],
)
def test_malformed_claimed_rows_fail_closed(record: dict[str, object]) -> None:
    with pytest.raises(TraceChunkRowCorruption, match="malformed trace chunk row"):
        decode_trace_storage_record(record)


def test_unsupported_claimed_row_format_fails_closed() -> None:
    with pytest.raises(TraceChunkRowFormatError, match="unsupported trace chunk"):
        decode_trace_storage_record({"storage_type": "trace-text-chunks-v2"})


@pytest.mark.parametrize(
    ("sequence0", "timestamp0", "timestamp_deltas", "reason"),
    [
        (2**63 - 2, 0, [0, 0], "sequence range"),
        (0, 2**63 - 1, [1, 0], "timestamp range"),
        (0, -(2**63), [-1, 0], "timestamp range"),
    ],
)
def test_claimed_rows_reject_reconstructed_int64_overflow(
    sequence0: int,
    timestamp0: int,
    timestamp_deltas: list[int],
    reason: str,
) -> None:
    record = {
        "storage_type": "trace-text-chunks-v1",
        "segment_id": "segment-a",
        "sequence0": sequence0,
        "timestamp0_us": timestamp0,
        "data": {
            "event_ids": ["a", "b", "c"],
            "turn_id": "turn-a",
            "call_id": "call-a",
            "block_id": "block-a",
            "timestamp_deltas_us": timestamp_deltas,
            "payloads": ["a", "b", "c"],
        },
    }

    with pytest.raises(TraceChunkRowCorruption, match=reason):
        decode_trace_storage_record(record)


def test_bad_row_never_returns_a_valid_prefix() -> None:
    good = _delta(0, "ordinary")
    malformed = {"storage_type": "trace-reasoning-chunks-v1"}

    with pytest.raises(TraceChunkRowCorruption):
        decode_trace_storage_records((good, malformed))


def test_invalid_or_extended_stream_shapes_lose_compression_not_data() -> None:
    invalid = [
        _delta(0, "a"),
        _delta(1, "b"),
        _delta(2, "c"),
    ]
    invalid[0]["sequence"] = True
    invalid[1]["timestamp_us"] = 1.5
    invalid[2]["payload"] = b"c"

    rows = pack_trace_event_rows(invalid)

    assert rows == tuple(invalid)
    assert decode_trace_storage_records(rows) == tuple(invalid)
