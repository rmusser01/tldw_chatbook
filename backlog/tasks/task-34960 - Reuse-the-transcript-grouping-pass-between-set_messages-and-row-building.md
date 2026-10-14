---
id: TASK-34960
title: Reuse-the-transcript-grouping-pass-between-set_messages-and-row-building
status: Done
assignee: []
created_date: 2026-10-13 14:49
labels:
- performance
- ui-perf
- wave-2-streaming
dependencies: []
priority: medium
updated_date: 2026-10-13 17:06
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
group_console_transcript_messages runs twice per refresh over the same message list — once in set_messages (console_transcript.py:4393) and again in _transcript_rows (:6947) — doubling the O(N) transcript-length-proportional work on every refresh tick during streaming. The second pass could reuse the units/spans the first pass already cached.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Units computed in set_messages are reused by the row-builder path (single grouping walk per refresh)
- [x] No duplicate grouping walk in the same refresh cycle
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read set_messages grouping + _transcript_rows regrouping
2. Cache units/spans from set_messages and reuse in the row builder
3. Unit tests
4. Targeted run + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Scope correction from investigation: `_transcript_rows` regroups the FILTERED visible list (pruned/hidden-tail excluded), not the same list set_messages groups — its walk is over a subset and cannot reuse the full-list units. The actual same-list duplicate walks were `_ownership_by_message_id()` regrouping `self._messages` on every call from `_compute_tail_trim_start` / `_measured_message_groups` / `_compute_prunable_prefix` (the per-refresh prune/measure helpers), plus the ingest itself.
- Fix: `_units_for_current_messages()` caches the grouping tuple keyed on the identity of `self._messages` — sound because set_messages installs a fresh list per ingest and nothing mutates it in place. `set_messages` consumes the cache for span building, and `_ownership_by_message_id(None)` now serves a cached ownership dict built (via extracted `_ownership_from_units`) from the same units. One full-list grouping walk per ingest serves ingest + spans + all three prune/measure helpers; explicit-list callers remain uncached.
- Tests: new `Tests/UI/test_console_transcript_units_cache.py` (3 tests: one walk per list across units+ownership consumers, ownership mapping correctness from shared units, explicit lists uncached). 3/3 pass. Regression: `test_console_raw_cli_transcript.py` 17/17 + the two new Wave-2 suites — 30/30; `test_console_transcript_fence_throttle.py` still shows exactly its 26 pre-existing errors (unchanged baseline).
- ADR required: no — per-instance memoization with an identity token, no contract change.
