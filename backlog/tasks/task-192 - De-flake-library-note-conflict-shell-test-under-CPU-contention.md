---
id: TASK-192
title: De-flake library note-conflict shell test under CPU contention
status: Done
assignee: []
created_date: '2026-07-12 14:05'
labels:
  - follow-up
  - test-flake
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
test_library_shell_note_conflict_shows_overwrite_reload_and_keeps_user_text intermittently fails under concurrent CPU load (passes in isolation). Widen its polling/timeout budget or replace the timing loop with condition-based waiting. Discovered during task 159; unrelated to that diff.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

Replaced the fixed-iteration (`for _ in range(150): ... else: raise`) conflict polls across the 7 note-conflict tests in `Tests/UI/test_library_shell.py` with a deadline-based `_wait_for_condition(pilot, predicate, *, timeout=15.0, message, interval=0.02)` helper: it checks the predicate first and returns immediately (no trailing pause) once true, otherwise polls on a wall-clock `time.monotonic()` deadline instead of a fixed iteration count, so the wait survives CPU contention. `message` accepts a plain string or a zero-arg callable evaluated at raise time (used by the two "falls back to list" tests so the diagnostic reports the actually-stuck state). Added 3 unit tests (`test__wait_for_condition_*`) against a minimal `_FakePilot` covering: immediate-true (checked once, no pause), timeout-raises-with-message, and callable-message-evaluated-at-raise. Converted all 12 conflict-family loops across the 7 named tests; left the bare `for _ in range(10): await pilot.pause(0.02)` settle-drain in the preview test and the ~91 other-feature `range(150)`/`range(N)` loops elsewhere in the file untouched (out of scope). All 10 tests (7 note-conflict + 3 helper) pass; confirmed via TDD RED (NameError) -> GREEN, then per-test conversion with the full gate green throughout.