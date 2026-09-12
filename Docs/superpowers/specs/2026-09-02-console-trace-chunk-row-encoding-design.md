# Console Trace Chunk-Row Encoding Design

**Task:** TASK-24206

**Status:** Approved by the existing task acceptance criteria and ADR-097 boundary

**Reference:** DeepSeek Harness `chunk-rows.ts`

## Problem

Future token-level Console traces can produce hundreds of consecutive text,
reasoning, or tool-call argument deltas. Persisting every delta as a complete
JSON event row repeats the same segment, turn, call, block, and type envelope
and makes physical storage scale with envelope overhead rather than payload.

The semantic trace ledger already defines one canonical logical event history.
Packing must remain invisible above persistence: readers receive the same
ordinary events in the same order, and existing unencoded records remain valid.

## Decision

Add a pure, versioned codec in `tldw_chatbook.Chat.console_trace_chunk_rows`.
It accepts JSON-compatible canonical event mappings and emits storage records.
Only an exact `stream_delta` event shape with a text, reasoning, or tool-call
delta may pack. Events with extra fields, invalid primitive types, unsupported
kinds, non-consecutive sequence values, or changed segment/turn/call/block
identity remain verbatim.

A compatible run of at least three events becomes one physical row tagged with
a storage-only `trace-*-chunks-v1` value. The row stores the first sequence and
timestamp, exact event identities and payload boundaries, and timestamp gaps.
The tag is not an event type and never escapes decoding.

Encoding is stateless per persistence batch. A run split across flushes may
produce more rows than the same run encoded in one batch, but concatenating the
decoded batches always reproduces the original history exactly. This preserves
durability without making callers retain or coordinate partial runs.

The decoder passes records without a storage tag through unchanged. A record
that claims any chunk-row storage tag must use the supported v1 tag and validate
its exact envelope, member arity, identities, integer bounds, and reconstructed
sequence/timestamp range. Unsupported formats raise a format diagnostic;
malformed supported rows raise a corruption diagnostic. Neither path yields a
partial history.

## Canonical stream-delta shape

Canonical events use exactly these fields:

- `event_id`, `segment_id`, `turn_id`, `call_id`, and `block_id`: non-empty strings
- `event_type`: `stream_delta`
- `delta_kind`: `text`, `reasoning`, or `tool_call`
- `sequence`: a non-negative signed-64-bit integer
- `timestamp_us`: a signed-64-bit integer
- `payload`: the exact string delta, including empty strings and Unicode

Exact-key recognition is deliberate. A future logical field or event variant
loses compression until the codec is revised, never data.

## Benchmark

The benchmark writes the same representative event stream to SQLite once as
ordinary JSON rows and once through the codec. It reports logical event count,
physical row count, allocated database bytes, encoded payload bytes, encode
time, and decode time. Tests require exact reconstruction plus physical row and
byte reduction; timing is reported rather than treated as a cross-platform
hard threshold.

The 6,000-event reference run on macOS 15.6 arm64, Python 3.12.11, and SQLite
3.49.1 produced:

| Measurement | Ordinary rows | Packed rows |
| --- | ---: | ---: |
| Physical rows | 6,000 | 3 |
| JSON bytes | 1,564,655 | 182,547 |
| SQLite allocated bytes | 1,724,416 | 192,512 |
| Encode time | 8.702 ms | 5.955 ms |
| Decode time | 6.636 ms | 2.653 ms |

Run the repeatable benchmark with
`python -m Tests.Benchmarks.benchmark_console_trace_chunk_rows --events 6000`.
The test gate asserts exact reconstruction and row/byte reduction; it does not
assert these machine-specific timing values.

## ADR check

ADR required: no

ADR path: `backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md`

Reason: ADR-097 already chose a physical codec beneath one logical trace model.
No database schema, migration, ownership, privacy, or runtime boundary changes.
