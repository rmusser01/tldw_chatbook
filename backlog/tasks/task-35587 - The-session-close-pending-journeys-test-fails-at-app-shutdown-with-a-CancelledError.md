---
id: TASK-35587
title: The session-close pending journeys test fails at app shutdown with a CancelledError
status: In Progress
created_date: 2026-10-19 03:53
updated_date: '2026-10-19 14:50'
assignee:
- '@claude'
labels:
- tests
- flaky
- console
priority: medium
dependencies:
- TASK-34982
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_session_tab_close.py::test_session_close_pending_journeys runs in a required lane (UI Fast Lane (8) and the Console P0 gate). In run 39151121023 (attempt 1, 2026-10-18, chore/task-33132-libcollections-migration, a DB-module refactor) it failed with asyncio CancelledError while _verify_background_pending_close_releases_round_without_an_active_turn's app shut down in run_test: Textual's Widget._message_loop_exit, gathering the child tasks of ConsoleTranscript (#console-native-transcript) during the screen prune, raised it. No .cancel() in Textual's own modules targets a widget's message-loop task (they cancel timer, worker, Markdown-stream, command-search and Pilot-wait tasks, and a call_later handle), so a child's task, or the transcript's own, was most likely cancelled from outside Textual. A failure on unchanged code counts against a pull request in the merge queue, and two in a row evict it. Split from TASK-35443, whose 'worker manager is closed' failure had a different cause.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The task that is cancelled at shutdown, and what cancels it, is identified, in the test or the product
- [ ] #2 The test passes 20 consecutive pull-request runs in CI
- [x] #3 If the product was at fault, a test that fails deterministically without the fix pins it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the frameless shutdown CancelledError deterministically, through the path TASK-34982 describes.
2. Fix it with TASK-34982 (one PR) and pin it with a test that fails without the fix.
3. After that PR merges, count 20 consecutive pull-request runs of this test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-19 UTC, first look (not a cause yet). The traceback has no frame below ConsoleTranscript's own `await gather(...)` in _message_loop_exit, which runs in _process_messages' finally after the loop's `except CancelledError: pass`. On the venv's CPython 3.12.11 (scratch script, three cases): gather over a child task cancelled before it ran keeps the child's frame; cancelling the awaiting task keeps the child's frames; only a child task whose cancellation an earlier awaiter already consumed gives a frameless CancelledError, because Future._make_cancelled_error hands out the saved exception once and builds a fresh one after that. So a direct child of the transcript most likely had a task that was cancelled and already awaited before shutdown.

TASK-34982 describes such a path: the transcript poll paints inside its timer's task, and stopping the timer while a paint awaits a row removal cancels the removed rows' tasks. Textual's Timer._stop_all then awaits the timer task and swallows its CancelledError, and the removal's gather has already consumed each row's. If those rows are still the transcript's children, a later prune's gather gets a fresh, frameless CancelledError, as in the CI run. TASK-34789 already cited the same path (as TASK-34814, its id before renumbering) as a possible cause of the headless-delivery test's WaitForScreenTimeout, so it may have more than one symptom. That this is what happened here is inferred, not shown. Next step: TASK-34982's deterministic timer-stop test, then whether its fix covers this shutdown.

Locally on dev 69771c1811 the test passed 5 of 5 sequential runs (75-107 s each).

2026-10-19 UTC, cause (AC #1, in the product). The cancelled tasks are removed transcript rows' own message-pump tasks. The transcript poll is a set_interval timer and paints inside the timer's own task; App._prune hands AwaitRemove the removed rows' pump tasks and await_prune gathers them. When the screen's message loop ends at shutdown, Textual's Timer._stop_all cancels the poll's timer task before the screen prunes its children, and if a paint is removing rows at that moment the cancel reaches their pumps through the gather: their Unmount is cut short, they stay the transcript's children with cancelled tasks, and the transcript's prune gather then raises a bare CancelledError (the first gather had already consumed each row's saved error). TASK-34982's test reproduces exactly this, with the CI run's signature (CancelledError from _message_loop_exit, no frame below ConsoleTranscript's gather). That a paint was removing rows in 39151121023 at that moment is inferred from the matching mechanism and signature; its log names no timer.

Fix and pin (AC #3): PR #3167 (TASK-34982) shields every awaited removal the poll's sync reaches (finish_removal in UI/Navigation/surface_swap_guard.py). Its Tests/UI/test_console_transcript_poll_stop_removal.py::test_stopping_the_paint_timer_mid_removal_still_removes_the_row stops the poll-style timer in the middle of a row removal; on dev it fails, with this task's CancelledError at run_test shutdown.

AC #2 is open: count 20 consecutive pull-request runs of test_session_close_pending_journeys (UI Fast Lane (8) and the Console P0 gate) after #3167 merges.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
