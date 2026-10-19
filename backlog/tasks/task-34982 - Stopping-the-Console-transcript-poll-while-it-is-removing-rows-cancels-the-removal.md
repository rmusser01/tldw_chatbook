---
id: TASK-34982
title: Stopping the Console transcript poll while it is removing rows cancels the
  removal
status: Done
created_date: 2026-10-12 05:26
updated_date: '2026-10-19 05:42'
assignee:
- '@claude'
labels:
- console
- stability
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console's transcript poll runs its paint inside the timer's own task. When the timer is stopped while a paint is waiting for transcript rows to be removed, the cancellation reaches the tasks of the rows being removed. In a test this surfaces as a cancellation error at teardown (one Console blocked-send test failed 8 times in 137 local runs under heavy load, on the branch and on the code before it alike, and never in the saved CI logs; its file is not in the PR gate). The same stop happens in the running app when the user leaves the Console while a paint is in that state; what the user would see then was not checked. Found while investigating flaky tests for TASK-34789.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stopping the transcript poll during a row removal lets the removal finish or ends it without an error
- [x] #2 Leaving the Console while a paint is removing rows leaves the transcript consistent on return
- [x] #3 A test stops the timer during a removal deterministically and fails before the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the mechanism without the Console: a timer callback awaiting a child removal, the timer stopped mid-removal.
2. Reproduce it on a real ConsoleTranscript, the paint run as the poll runs it, and fail first.
3. Let a started removal finish whatever cancels its awaiter, at every awaited removal the poll's paint reaches.
4. Check the transcript is consistent on the next paint; compare the Console census files against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Cause (Textual 8.2.8). App._prune hands AwaitRemove the removed widgets' own message-pump tasks, and await_prune gathers them. The poll is a set_interval timer, which runs its callback inside the timer's own task (set_timer instead hands the callback to the target's pump with call_next). Stopping the timer cancels that task, and the cancel reaches the rows' pumps through the gather while they are exiting: their Unmount is interrupted, _message_loop_exit never detaches them, and they stay in the transcript with cancelled tasks. The transcript's next prune (app shutdown included) gathers them and raises a bare CancelledError, frameless because the first gather already consumed each row's saved error. Stops come from the poll itself, on_screen_suspend (leaving the Console), on_unmount, and Textual's Timer._stop_all when the screen's message loop ends, which runs before the screen prunes its children.

Fix: finish_removal() in UI/Navigation/surface_swap_guard.py (beside the other awaited-removal rule) awaits a removal through asyncio.shield, so the removal finishes on its own while the caller still sees its cancellation. Two edges: a caller that is itself one of the removed widgets' pumps awaits the removal directly, since AwaitRemove leaves the current task out of what it waits on and the shield's new task would not be left out (it would wait on itself); and a removal that fails after its caller was cancelled is logged at WARNING with its exception type only, so the failure is not lost.

It is used at the awaited removals the poll's sync reaches. They were found with an awaited-call graph over UI/, Widgets/ and Chat/console_*.py from _sync_native_console_chat_ui and sync_poll_display_ui, by name, plus the wired avatar renderer: ConsoleTranscript._reconcile_rows, _sync_assistant_turn_widget (adjunct stack), _sync_activity_widgets (activity, action and detail stacks), dismiss_message_more_menu (reached through refresh_messages when a message menu is open), ConsoleAssistantTurnWidget.replace_detail_widgets, the session tab strip's rebuild in ConsoleSessionSurface.sync_sessions, and LeftRail.replace_character_avatar_widget (both removals; reached through the sync's avatar refresh). The graph's other hits (the prompts and fork-chat modals) come only through name collisions on prepare and cancel, not a real path. Removals elsewhere are left as they are.

Evidence:
- Tests/UI/test_console_transcript_poll_stop_removal.py, new and in the UI PR gate (census floor 324 -> 325). The transcript test pins the _reconcile_rows site end to end; the three helper tests pin the contract every site relies on. The transcript test runs the paint as the poll does (set_messages, then await refresh_messages, inside a set_interval callback), holds the removed row's Unmount, stops the timer, then releases it. On dev it fails with 'the stopped paint left its removed row mounted' and then a CancelledError from _message_loop_exit at run_test shutdown, the signature of TASK-35587's CI failure; with the fix the row is removed, its task is not cancelled, and the next refresh mounts exactly the current rows (AC #2). The helper tests: a held removal still completes when its caller is cancelled; a widget awaiting its own removal from its own pump completes (it deadlocks, and the test times out, without the direct-await edge); a removal failing after its caller was cancelled is logged (nothing is logged without the done-callback). Each was red against the helper with that part removed.
- Plain Tests/UI files cannot run locally (the RecoveryRequired wall), so the four tests were run through a standalone runner under the scratch profile: the transcript test red on dev, and all four green on the final head.
- The UI census's 68 Console files, run whole on this branch and on dev 69771c1811 (xdist -n 8; most failures are the local wall). The first run gave 114 failed and 1314 passed here, 113 and 1315 on dev, the same set but for three. test_workspace_command_opens_same_manager_in_registered_scope failed on dev only. test_a_probe_settling_mid_poll_still_reaches_the_console (readiness text not yet refreshed) and test_console_r_resends_a_refused_echo_as_one_message (timed out waiting for the resend button, a flake listed under TASK-34789) failed here only, and both passed 3 of 3 serially on each tree. After review round 1 (three more sites, the direct-await and logging edges) the branch run gave 112 failed and 1316 passed, every failure also failing on dev.
- Tests/CI: 998 passed, 1 skipped. Preflight passes. Diagnostic inventory: one added call, the WARNING above; reviewed with --statements: it interpolates only an exception type name (no user content, secret, path or URL); regenerated.

Files: tldw_chatbook/UI/Navigation/surface_swap_guard.py, tldw_chatbook/Widgets/Console/console_transcript.py, console_assistant_turn.py, console_session_surface.py, tldw_chatbook/UI/Console_Modules/left_rail.py, Docs/security/production-diagnostic-inventory.json, Tests/UI/test_console_transcript_poll_stop_removal.py, scripts/ui_pr_gate_census.txt, scripts/check_ui_pr_gate_census.py.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A started DOM removal on the Console poll's sync path now finishes even when its caller is cancelled: stopping the poll's timer (leaving the Console, or the screen closing) no longer cancels the removed rows' own pumps, so no row is left mounted with a dead task and no bare CancelledError is raised at the next prune or at shutdown.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Renumbering provenance

Renumbered from task-34814 to TASK-34982 (2026-10-13): this ID collided
with a concurrently filed task after the branches merged; per the
2026-08-21 owner rule (TASK-19601) the older arrival kept the id.
