---
id: TASK-34924
title: Replace-per-chunk-json.dumps-trace-sizing-with-a-running-byte-counter
status: Done
assignee: []
created_date: 2026-10-13 14:50
labels:
- performance
- ui-perf
- wave-2-streaming
dependencies: []
priority: low
updated_date: 2026-10-13 16:56
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The trace accumulator measures response bytes per chunk by fully serializing an envelope with json.dumps(sort_keys=True) on the event loop (console_provider_gateway.py:2313-2335 -> :2498-2513), until roughly 768KB has accumulated — i.e. for the entire length of most replies. The encoded length of the item (or a running counter) is O(1)-ish per chunk.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Trace byte accounting uses item encoded length or a running counter without full envelope re-serialization per chunk
- [x] Reported byte totals match the previous measurement within documented tolerance (test)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read observe_response + _trace_response_item_bytes and the byte accounting contract
2. Measure encoded item length without full envelope re-serialization
3. Targeted test run + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- `_trace_response_item_bytes` (console_provider_gateway.py) now fast-paths the two per-token item shapes — plain content deltas and ProviderThinkingDelta (reasoning models stream thinking per token too) — through `_json_escaped_string_bytes`: UTF-8 length + one-byte-per-escape counts (quote/backslash/newline/CR/tab, the complete set for ensure_ascii=False besides rare C0 controls) + a computed envelope-wrapper allowance (the literal `{"content":…,"role":"assistant"}` lengths for deltas; fixed keys + the item's short metadata strings for thinking evidence). No envelope construction, no `json.dumps`, no deepcopy per chunk. Structured items (tool calls etc.) are rare per response and keep the exact envelope serialization.
- Documented approximation: exotic C0 control characters would undercount by ≤5 bytes each — immaterial for the accumulation bound (the limit exists to bound retained trace size, not to account exactly).
- Tests: new `Tests/Chat/test_console_trace_item_bytes.py` (16 tests) — the cheap string length is EXACTLY equal to `len(json.dumps(text, ensure_ascii=False).encode())` across ASCII/quotes/newlines/unicode/emoji samples, and item totals stay within 16/64 bytes of the full-envelope measurement (accumulated stream total within 5%).
- Regression: `Tests/Chat/test_console_trace_settlement.py` 33/33+3 pre-existing failures — identical with the change stashed; gateway `-k "trace or accumulation"` selection 5/5.
- ADR required: no — measurement internals of one accumulator.
