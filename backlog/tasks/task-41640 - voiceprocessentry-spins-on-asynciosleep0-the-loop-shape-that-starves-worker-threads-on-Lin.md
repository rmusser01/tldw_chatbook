---
id: TASK-41640
title: "voice_process_entry spins on asyncio.sleep(0), the loop shape that starves worker threads on Linux"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- audio
- flaky
priority: low
references:
- tldw_chatbook/Audio/voice_process_entry.py
- backlog/docs/lessons-testing-evidence.md
dependencies:
- TASK-41317
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-40334 showed that an `await asyncio.sleep(0)` polling loop keeps the event loop on-CPU on Linux and starves threads waiting for the GIL. TASK-41317 removed one such loop (the relay publisher's `wait_idle`). `Audio/voice_process_entry.py:186` has the same shape.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The loop waits on an event, queue or future instead of re-queuing itself
- [x] #2 A test counts loop turns during the wait (as TASK-41317's did) and fails on the old code
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (a private wait inside the audio child's close changes how it blocks; no contract, behaviour or architecture changes).

1. Test first in `Tests/Audio/test_voice_process_core.py`: a prepared `_ProductionSession` whose core holds its close lock, so resource cleanup has not started; count real loop turns (`_run_once` wrapper, as TASK-41317's test) while `begin_close()`'s resources wait runs for 0.2 s; assert a bounded count, then release the lock and assert the wait returns `CLEAN`. Show it red on the unfixed code.
2. Fix: the core sets an `asyncio.Event` when it creates `_resource_cleanup_task`; `begin_close`'s `resources()` awaits that event instead of polling with `sleep(0)`.
3. Mutation check, covering Audio voice-process files on base and head, preflight, `Tests/Architecture` serial.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`_ProductionSession.begin_close`'s `resources()` no longer polls. The core (`ConsoleSpeculativeVoiceSession`) sets a new `_resource_cleanup_started` `asyncio.Event` right after `_close_once` creates `_resource_cleanup_task`, and `resources()` awaits that event before shielding the task. If close fails before cleanup starts, the wait now sleeps rather than spins until the child's existing `close_deadline` reports `resources_closed: failed`; the result is the same as before.

- Test: `test_production_child_close_lets_an_idle_loop_sleep_until_cleanup_starts` (`Tests/Audio/test_voice_process_core.py`). It prepares a real `_ProductionSession` (the existing test's fakes, now shared through `_production_session`), holds the core's `_close_lock` so cleanup cannot start, and counts real loop turns (a `_run_once` wrapper, as TASK-41317's test does) over 0.2 s while the resources wait is pending. Then it releases the lock and asserts the wait returns `CLEAN`. It measured 3219 turns on the unmodified code and 3 after the fix. Reverting only `voice_process_entry.py` gives 653 turns and a failure.
- Covering files (15 files that import `voice_process_core` or `voice_process_entry`, `-n 4`, load 40-90): base d0486978dd 106 failed / 361 passed, head 108 failed / 360 passed (+1 new test). The failures are pre-existing (`test_console_voice_preflight` "unexpected" preparation errors and others). Three failed only on head; run alone, two passed, and `test_first_stage_callbacks_are_wired_through_real_attempt_and_sequencer` is flaky on both (base 3 of 3 failed, head 2 of 3).
- `./scripts/preflight.sh` passes. `Tests/Architecture` (serial, load 40-90): 4 failed, 1475 passed, 5 skipped. Three also fail on base d0486978dd: `test_reviewed_diagnostic_changes_are_metadata_only` and the two `test_timer_path_static_update_inventory.py` tests. The fourth, `test_task_15103_review_ledger_canonical_schema_and_arithmetic`, failed only in the full run and passes alone on head (136 s), the same load timeout TASK-41317 recorded.
- Files: `tldw_chatbook/Audio/voice_process_core.py`, `tldw_chatbook/Audio/voice_process_entry.py`, `Tests/Audio/test_voice_process_core.py`.
<!-- SECTION:NOTES:END -->
