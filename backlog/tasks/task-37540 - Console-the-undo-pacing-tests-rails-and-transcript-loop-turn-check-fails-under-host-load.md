---
id: TASK-37540
title: >-
  Console: the undo pacing test's rails-and-transcript loop-turn check fails
  under host load
status: To Do
assignee: []
created_date: '2026-10-29 02:40'
labels:
  - console
  - testing
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during TASK-33620.15.20's by-id suite comparison (2026-10-29). `Tests/UI/test_console_undo_restore_pacing.py::test_a_console_sync_runs_its_rails_and_its_transcript_in_separate_loop_turns` pins TASK-33628.5.1's rule that a whole Console sync yields between its rails half and its transcript half. On a detached origin/dev 3efe8476ce it failed 3 of 3 runs at host load 29-34, with "the transcript half ran in the rails half's loop turn" and a first observation of False each time. The branch under test failed the same way, and dev had passed the test twice earlier that night at other loads.

The test wraps the screen's `_sync_console_workspace_context` (named on 46 lines across the app), and pairs the latest rails call with the next transcript call. So the red has one of two causes, and the test cannot tell them apart:
- some sync path runs its transcript half in the same loop turn as a rails sync, so the split is incomplete;
- a rails call from another path is paired with this sync's transcript half.

Either way, a pin that guards UI-loop pacing currently depends on host load.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the first False observation under load is named on origin/dev: which caller's rails sync preceded the transcript half that observed it
- [ ] #2 If a Console sync path runs its transcript half in the rails half's loop turn, that path yields between the two; otherwise the test pairs each transcript half only with the rails half of the same sync
- [ ] #3 The test passes 10 of 10 runs at host load above 25, and still fails when the sync's yield between its two halves is removed
<!-- AC:END -->
