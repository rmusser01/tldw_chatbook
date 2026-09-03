"""Representative storage benchmark for Console trace chunk rows."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import json
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import TypedDict

from tldw_chatbook.Chat.console_trace_chunk_rows import (
    decode_trace_storage_records,
    pack_trace_event_rows,
)


class ChunkRowBenchmarkResult(TypedDict):
    """Measurements for one raw-versus-packed benchmark run."""

    logical_events: int
    raw_rows: int
    packed_rows: int
    raw_json_bytes: int
    packed_json_bytes: int
    raw_database_bytes: int
    packed_database_bytes: int
    raw_encode_ms: float
    packed_encode_ms: float
    raw_decode_ms: float
    packed_decode_ms: float


def representative_events(event_count: int) -> list[dict[str, object]]:
    """Build a deterministic stream covering every supported delta kind."""

    if event_count < 9:
        raise ValueError("event_count must be at least 9")
    kinds = ("text", "reasoning", "tool_call")
    events: list[dict[str, object]] = []
    for sequence in range(event_count):
        kind_index = min(sequence * len(kinds) // event_count, len(kinds) - 1)
        kind = kinds[kind_index]
        events.append(
            {
                "event_id": f"event-{sequence:08d}",
                "segment_id": "segment-benchmark",
                "sequence": sequence,
                "event_type": "stream_delta",
                "timestamp_us": 1_800_000_000_000_000 + (sequence * 1_137),
                "turn_id": "turn-benchmark",
                "call_id": "call-benchmark",
                "block_id": f"block-{kind}",
                "delta_kind": kind,
                "payload": ("token-🙂" if sequence % 17 == 0 else "token"),
            }
        )
    return events


def _serialize(
    records: Iterable[Mapping[str, object]],
) -> tuple[list[str], float]:
    started = time.perf_counter_ns()
    rows = [
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in records
    ]
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return rows, elapsed_ms


def _decode(rows: list[str]) -> tuple[tuple[object, ...], float]:
    started = time.perf_counter_ns()
    decoded = decode_trace_storage_records(json.loads(row) for row in rows)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return decoded, elapsed_ms


def _write_database(path: Path, rows: list[str]) -> int:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE trace_records (position INTEGER PRIMARY KEY, body TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO trace_records(body) VALUES (?)",
            ((row,) for row in rows),
        )
        connection.commit()
    finally:
        connection.close()
    return path.stat().st_size


def run_benchmark(event_count: int = 6_000) -> ChunkRowBenchmarkResult:
    """Compare raw event rows with lossless packed rows in fresh SQLite files."""

    events = representative_events(event_count)
    raw_rows, raw_encode_ms = _serialize(events)

    packed_started = time.perf_counter_ns()
    packed_records = pack_trace_event_rows(events)
    packed_rows = [
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in packed_records
    ]
    packed_encode_ms = (time.perf_counter_ns() - packed_started) / 1_000_000

    raw_decoded, raw_decode_ms = _decode(raw_rows)
    packed_decoded, packed_decode_ms = _decode(packed_rows)
    expected = tuple(events)
    if raw_decoded != expected or packed_decoded != expected:
        raise AssertionError("benchmark storage did not reconstruct the exact trace")

    with tempfile.TemporaryDirectory(prefix="trace-chunk-row-benchmark-") as directory:
        benchmark_dir = Path(directory)
        raw_database_bytes = _write_database(benchmark_dir / "raw.sqlite3", raw_rows)
        packed_database_bytes = _write_database(
            benchmark_dir / "packed.sqlite3", packed_rows
        )

    return {
        "logical_events": len(events),
        "raw_rows": len(raw_rows),
        "packed_rows": len(packed_rows),
        "raw_json_bytes": sum(len(row.encode("utf-8")) for row in raw_rows),
        "packed_json_bytes": sum(len(row.encode("utf-8")) for row in packed_rows),
        "raw_database_bytes": raw_database_bytes,
        "packed_database_bytes": packed_database_bytes,
        "raw_encode_ms": raw_encode_ms,
        "packed_encode_ms": packed_encode_ms,
        "raw_decode_ms": raw_decode_ms,
        "packed_decode_ms": packed_decode_ms,
    }


def main() -> None:
    """Run the benchmark and print machine-readable JSON."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=6_000)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.events), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
