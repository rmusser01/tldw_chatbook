"""Release checks for the Console trace chunk-row benchmark."""

from Tests.Benchmarks.benchmark_console_trace_chunk_rows import run_benchmark


def test_chunk_rows_reduce_representative_storage_without_losing_events() -> None:
    result = run_benchmark(event_count=1_200)

    assert result["logical_events"] == 1_200
    assert result["raw_rows"] == 1_200
    assert result["packed_rows"] == 3
    assert result["packed_json_bytes"] < result["raw_json_bytes"]
    assert result["packed_database_bytes"] < result["raw_database_bytes"]
    assert result["raw_encode_ms"] >= 0
    assert result["packed_encode_ms"] >= 0
    assert result["raw_decode_ms"] >= 0
    assert result["packed_decode_ms"] >= 0
