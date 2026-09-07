# TASK-24206 Console Trace Chunk-Row Encoding Plan

## Goal

Provide a lossless, versioned storage-row codec for future streamed Console
trace deltas without changing the canonical logical event model or current
capture/runtime wiring.

## Implementation

1. Add focused red tests in `Tests/Chat/test_console_trace_chunk_rows.py` for
   exact round trips, compatibility boundaries, split persistence batches,
   malformed/unsupported rows, legacy pass-through, Unicode, empty chunks,
   timestamp gaps, interrupted streams, short runs, and mixed kinds.
2. Implement the smallest strict codec in
   `tldw_chatbook/Chat/console_trace_chunk_rows.py`, using exact-key admission,
   three storage-only v1 tags, and explicit corruption/format exceptions.
3. Add `Tests/Benchmarks/benchmark_console_trace_chunk_rows.py` and its focused
   test to compare SQLite row count/bytes and encode/decode cost against
   unencoded JSON rows.
4. Record the representative benchmark and format contract in the user-facing
   trace documentation, then update the task acceptance criteria and concise
   implementation notes.
5. Run the focused codec/benchmark tests, relevant semantic-trace regression
   tests, Ruff, Python compilation, and `git diff --check`.

## Review risks

- Physical storage tags must never be accepted as logical event types.
- Unknown or extended logical events must pass through verbatim.
- A malformed claimed row must never decode a valid prefix.
- Sequence and timestamp delta arithmetic must remain within signed 64-bit
  bounds so JSON/SQLite round trips are deterministic.
- Persistence batch shape may affect compression ratio but not decoded history.

## ADR check

ADR required: no

ADR path: `backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md`

Reason: this is the direct physical-codec follow-up already separated by
ADR-097 and does not change durable schema or ownership.
