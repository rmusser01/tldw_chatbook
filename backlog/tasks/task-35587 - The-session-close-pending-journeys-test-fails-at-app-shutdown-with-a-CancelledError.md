---
id: TASK-35587
title: The session-close pending journeys test fails at app shutdown with a CancelledError
status: In Progress
created_date: 2026-10-19 03:53
updated_date: '2026-11-23 07:09'
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
- [x] #1 The task that is cancelled at shutdown, and what cancels it, is identified, in the test or the product (mechanism reproduced; the CI trigger is inferred, and AC #2 confirms it)
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

2026-10-19 UTC, cause (AC #1, in the product): the TASK-34982 path, whose notes have the detail. At app shutdown the Console screen's teardown stops the transcript poll's set_interval timer before the screen's children are gathered; a paint removing rows at that moment has its cancel reach the rows' own pump tasks, they stay attached with cancelled tasks, and ConsoleTranscript's gather raises a frameless CancelledError (an earlier gather had consumed each row's saved error). That a paint was removing rows in 39151121023 is inferred from the matching mechanism and signature; its log names no timer.

Fix and pin (AC #3): PR #3167 (TASK-34982). Its test_stopping_the_paint_timer_mid_removal_still_removes_the_row fails on dev with this task's CancelledError at run_test shutdown.

AC #2 is open: 20 consecutive pull-request runs of test_session_close_pending_journeys (UI Fast Lane (8), Console P0 gate) after #3167 merges.

2026-11-23 (round-22 filing): CI history and timing. Across every attempt of the 321 Derived Artifacts runs created from 2026-11-19 06:00Z to 2026-11-23 06:00Z, the UI Fast Lane job that runs this file completed 157 times. The first page of nine runs' job lists (44662627953, 44871429295, 44909027257, 44922739689, 44929982891, 44954395594, 44977790794, 45016281720, 45034674868) returned Server Error at 100 jobs per page and was re-read at 30 per page; it held 13 more completed jobs. This test failed in 29 of them, and every one was `child failure: AssertionError: closing the background tab answered the viewed tab`. That is TASK-42424's cause (round 22); its fix is in the tests and was not on the branches that failed. The 26 failures found on the first read fall between 2026-11-20 09:06Z and 2026-11-23 05:46Z; the three found on the re-read are run 44909027257 attempt 2 (job 136862152801) and run 45016281720 attempts 2 and 3 (jobs 137190507806, 137216650072). None was a timeout: the test's call time was about 30-84 s (the first 26 failures' median 48 s, passing runs median 50 s) against the lane's 180 s `--timeout`. Locally, TASK-42424's implementer measured about 131 s serially at load 52, so the headroom is much smaller on a loaded machine. TASK-42426's third review saw one more local failure at load 26-50 whose reason was not captured; the test passed 3 times after. AC#2's 20 consecutive pull-request runs should be counted from TASK-42424's merge. A failure with another message, or a call time near 180 s, is new evidence for this task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
