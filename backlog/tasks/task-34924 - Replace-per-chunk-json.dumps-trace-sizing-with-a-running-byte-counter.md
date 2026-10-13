---
id: TASK-34924
title: Replace-per-chunk-json.dumps-trace-sizing-with-a-running-byte-counter
status: To Do
assignee: []
created_date: '2026-10-13 14:50'
labels:
  - performance
  - ui-perf
  - wave-2-streaming
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The trace accumulator measures response bytes per chunk by fully serializing an envelope with json.dumps(sort_keys=True) on the event loop (console_provider_gateway.py:2313-2335 -> :2498-2513), until roughly 768KB has accumulated — i.e. for the entire length of most replies. The encoded length of the item (or a running counter) is O(1)-ish per chunk.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Trace byte accounting uses item encoded length or a running counter without full envelope re-serialization per chunk,Reported byte totals match the previous measurement within documented tolerance (test)
<!-- AC:END -->
