---
id: TASK-1230
title: Navigation-guard dialog can wedge into an app soft-lock
status: Done
assignee:
  - '@claude'
created_date: '2026-07-28 09:30'
updated_date: '2026-07-28 16:40'
labels:
  - console
  - navigation
  - critical
  - uat
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expert UAT (Docs/superpowers/qa/fleet-ux-expert-review-2026-07-28, F1): with a busy fleet, navigate (guard dialog) -> Stay (works) -> navigate again -> click Leave at its rendered coordinates -> no effect, and thereafter the dialog answers to nothing (both buttons via 12-point click sweep, Escape, Tab, Enter, nav-bar clicks all inert). Only Ctrl+Q escapes. App log empty; mechanism undetermined (hypotheses in the report: post-confirm navigation failure leaving a painted-but-dead overlay; or a push_screen_wait interleaving the existing race test does not cover — that test queues a second NavigateToScreen message, not a Stay-then-renavigate-then-Leave human sequence). Task-1142 rhyme: the guard's tests click the button widget; nothing clicks rendered coordinates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The scripted repro (Stay, renavigate, Leave-by-coordinates) navigates cleanly with runs cancelled; no input-inert state is reachable.
- [x] #2 The dialog is keyboard-operable (documented keys) and Escape maps to Stay.
- [x] #3 A coordinate-honest regression test drives the full human sequence at rendered positions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce headlessly with real driver-style MouseDown/MouseUp events (not pilot.click, which bypasses App.on_event) to confirm the mechanism.
2. Diagnose: TldwCli.handle_screen_navigation is an @on(NavigateToScreen) handler awaited inline on the App's own single message-processing task; ChatScreen.confirm_navigation's push_screen_wait-via-worker pattern only satisfies the worker-context requirement for the wait itself, it does not decouple the WAIT from that task, so the task stays suspended for the dialog's entire lifetime -- starving App.on_event, the only path real driver-originated clicks/keys use to reach the dialog.
3. Fix: split handle_screen_navigation into a thin @on(NavigateToScreen) dispatcher (_dispatch_screen_navigation) that fires the existing handle_screen_navigation body as its own worker and returns immediately, freeing the App's message loop; add an asyncio.Lock to preserve FIFO ordering across overlapping navigation attempts since they're no longer serialized by the single message queue.
4. Add keyboard-hint copy to the dialog message (Tab/Shift+Tab/Enter/Esc); verify Escape-maps-to-Stay already works (ConfirmationDialog's existing binding).
5. Write a coordinate-honest regression test driving the full Stay-then-renavigate-then-Leave sequence via raw MouseDown/MouseUp through app.post_message (mirrors textual.driver.Driver.send_message), revert-verified RED against pre-fix code.
6. Run the full test_screen_navigation.py + test_console_parallel_runs.py suite; fix the one legitimate timing regression surfaced (rapid-tab-switch-storm test's zero-settle assumption, now stale given navigation is genuinely async).
7. Verify live in a real tmux-driven TUI session against a running llama_cpp provider with a real agent run.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Diagnosed mechanism (verified by direct reproduction, not just theory): TldwCli.handle_screen_navigation was an @on(NavigateToScreen) handler, and Textual awaits every such handler INLINE on the App's own single message-processing task (_process_messages_loop -> _dispatch_message). ChatScreen.confirm_navigation's busy-fleet dialog uses self.run_worker(push_screen_wait(dialog)) + await worker.wait() -- this satisfies push_screen's get_current_worker() requirement, but the AWAIT of that worker still happens on handle_screen_navigation's own call stack, i.e. on the App's message-processing task. Real driver-originated MouseDown/MouseUp/Key events ALSO land on that same task's queue (App.on_event is dispatched the identical way) and cannot be dequeued while the task is suspended awaiting the dialog -- confirmed directly by posting raw MouseDown+MouseUp via app.post_message (mirroring textual.driver.Driver.send_message; pilot.click() cannot be used for this, it bypasses App.on_event entirely via screen._forward_event) and observing the queue grow without ever draining. This affects EVERY confirm-navigation dialog opened this way, not just a 'second' one; the live human repro's dialog #1 apparently working was not reproducible as a guaranteed behavior once tested with true driver-style events -- both dialogs are equally exposed.

Fix: split handle_screen_navigation into a thin @on(NavigateToScreen) dispatcher, _dispatch_screen_navigation, that runs the existing (unmodified) handle_screen_navigation body as its own worker via self.run_worker(...) and returns immediately -- freeing the App's message-processing task to keep routing input the moment any confirm dialog opens. handle_screen_navigation itself is unchanged and remains directly awaitable to completion, so the ~30 existing tests that call it directly (bypassing the message queue) are unaffected. Added _screen_navigation_lock (asyncio.Lock) to preserve the old single-queue FIFO ordering across navigation attempts, since they are no longer serialized by a shared message queue.

Also added a keyboard-hint line to the dialog's own message copy ('Tab/Shift+Tab selects Stay or Leave, Enter activates the selected button, Esc stays.') -- Escape-maps-to-Stay, Tab-cycles-focus, and Enter-activates all already worked via Textual/ConfirmationDialog defaults; this only documents them per AC#2.

New coordinate-honest regression test: Tests/UI/test_console_parallel_runs.py::test_navigation_guard_survives_stay_then_renavigate_then_leave_by_coordinates. Drives the exact human sequence (busy fleet -> dialog #1 -> real click at Stay's rendered coordinates -> renavigate -> dialog #2 -> real click at Leave's rendered coordinates) using raw MouseDown/MouseUp posted through app.post_message, NOT pilot.click (which would hide this exact bug class, echoing the task-1142 test-blindspot). Revert-verified RED: reverting app.py/chat_screen.py against this test reproduces textual.app.ScreenStackError: No screens on stack during teardown (the queued click was never processed while the message pump was stuck).

Ran and fixed a real, measurable regression surfaced by the full gate suite: test_rapid_tab_switch_storm_leaves_no_zombie_widgets posts 12+1 NavigateToScreen messages with zero pacing then asserts zero zombie widgets the instant the final target screen and its is_running flag first report ready. With navigation now genuinely decoupled via workers (each queueing behind _screen_navigation_lock with real worker-scheduling overhead), the backlog of 13 stacked attempts takes a legitimate ~200ms longer to fully settle than the test's single-tick check allowed for -- confirmed via direct benchmarking that a REALISTIC single idle-fleet navigation's latency is unchanged (even slightly faster in this run), so 'idle-fleet navigation stays instant' is preserved; only the artificial zero-pacing storm needed a bounded settle-and-recheck loop instead of a one-shot assertion (still fails correctly if the app ever regresses to the historical stuck-widget/instance-cache bug this test guards against).

Live-verified in a real tmux TUI (235x52) against the actually-running llama_cpp provider at 127.0.0.1:9099: started a real agent run (which spawned sub-agents), navigated away (dialog #1), clicked Stay at its rendered screen coordinates (real SGR mouse click) -- dialog closed, run continued; renavigated, clicked Leave at its rendered coordinates -- navigation completed cleanly to the target screen, run torn down; separately verified Escape (real key press) maps to Stay on a fresh dialog. App remained fully responsive throughout (further navigations, a second run) with no soft-lock. Scratch profile and tmux server cleaned up after.

Modified: tldw_chatbook/app.py (_dispatch_screen_navigation, _screen_navigation_lock, handle_screen_navigation split), tldw_chatbook/UI/Screens/chat_screen.py (dialog copy + comment), Tests/UI/test_console_parallel_runs.py (new regression test), Tests/UI/test_screen_navigation.py (settle-tolerant zombie-widget check in test_rapid_tab_switch_storm_leaves_no_zombie_widgets).
<!-- SECTION:NOTES:END -->
