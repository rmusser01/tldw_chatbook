---
id: TASK-34959
title: Make-streaming-fence-detection-O-delta-instead-of-a-full-body-rescan
status: Done
assignee: []
created_date: 2026-10-13 14:48
labels:
- performance
- ui-perf
- wave-2-streaming
dependencies: []
priority: medium
updated_date: 2026-10-13 17:02
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On every 0.2s sync tick for every streaming markdown row, the open-fence check (console_transcript.py:1946 -> :1248-1313) splits the ENTIRE accumulated body into a list and regex-matches line-by-line — even for replies that contain no fence at all (the scan is how it finds out). Quadratic per message over the stream; the docstring itself calls the cost unbounded with message size. A running fence-parity counter updated from the appended delta would make it O(delta).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Fence parity tracked incrementally from appended deltas or scanning limited to the unexamined tail
- [x] No per-tick full-body split+regex over unchanged text
- [x] Open-fence flush behavior identical (existing fence tests pass)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read _console_markdown_body_ends_in_open_fence + call sites
2. Track fence parity incrementally (scan only the unseen tail)
3. Unit tests for parity across append/flush
4. Targeted run + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- New `_StreamingFenceParity` (console_transcript.py): complete lines fold exactly once (`scanned_upto` sits just past the last consumed newline — the append-only growth branch of `sync_message` guarantees the prefix is stable), and only the trailing partial line — which a later delta can turn into a different delimiter line — is judged fresh per call without committing. Per streamed tick the work is now the delta's new lines plus one partial-line regex, not the whole body.
- `_console_markdown_body_ends_in_open_fence` remains as the one-shot whole-body verdict (existing tests import it) and delegates to the same fold rules, so the two can never drift.
- Guard rails: the widget's non-append replace branch resets the tracker alongside the pending-delta discard; the tracker itself additionally self-resets when the consumed-prefix invariant breaks (`scanned_upto` beyond body length, or the char before `scanned_upto` is no longer a newline) — O(1) detection of any non-append change that skipped reset().
- Tests: new `Tests/UI/test_console_streaming_fence_parity.py` (10 tests) — per-step verdict equality with the one-shot scan across seven tricky chunk streams (split fences, tilde fences, backtick-in-info-string, empty deltas), fold-once `scanned_upto` accounting, shrink/invariant-break self-reset, partial-tail rejudgment. All pass.
- NOTE: `Tests/UI/test_console_transcript_fence_throttle.py` shows 26 setup errors at this branch's HEAD with the change stashed (same pre-existing mount breakage); parity is covered by the new self-contained tests.
- ADR required: no — internal algorithm swap with pinned verdict equivalence.
