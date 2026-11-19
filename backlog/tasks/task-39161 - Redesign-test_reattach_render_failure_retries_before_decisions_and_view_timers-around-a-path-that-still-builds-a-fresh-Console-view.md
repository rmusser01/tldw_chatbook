---
id: TASK-39161
title: "Redesign test_reattach_render_failure_retries_before_decisions_and_view_timers around a path that still builds a fresh Console view"
status: Done
assignee: ['@claude']
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 17:36
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_store_continuity.py
- tldw_chatbook/UI/Screens/chat_screen.py
dependencies:
- TASK-39120
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39120 on round-12 branch fu/r12-ctests). Checked on the round-12 stack 9091820bf2, 2026-11-08.

- The test needs a fresh Console view to re-attach after navigation: it checks that a failed re-attach render retries before the decision and view timers.
- Since fed0b7257a (TASK-31520, 2026-09-05), the Console route is reused. Navigating back resumes the same screen, so no view starts and no `_restore_collapsible_states` timer is set.
- TASK-39120's notes classify the red as test drift from that commit. They propose a redesign "at a path that still builds a fresh Console (the reusable-route cache dropped on a runtime-identity flip)".
- On the stack, under a temporary `bootstrap_profile` marker (scratch plugin, run.sh, `-n 4`, load ~25), the node fails `assert [] == [True]`. The file's other five nodes pass. Plain, the file stops at `RecoveryRequired`.
- No PR lane runs the file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test drives a path that builds a fresh Console view (for example a runtime-identity flip that drops the route cache) and still proves its intent: a failed re-attach render retries before the decision and view timers
- [x] #2 With the retry temporarily removed, the test is red; with it restored, green
- [x] #3 The node passes 3 of 3 runs at `-n 4`, with load recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Find a production path that still builds a fresh Console view: `_reusable_navigation_screen` drops the cached instance on a runtime-identity flip.
2. Flip the identity (the app's `_current_runtime_identity` seam) between leaving Console and returning, and assert the return built a new `ChatScreen`.
3. Keep the test's intent: inject the failure only into the attach reconciliation's own sync, assert no decision remount or view timer before the successful retry, and both after.
4. Negative control with the reconciliation retry removed; 3 runs at `-n 4` under a temporary `bootstrap_profile` marker (RecoveryRequired trap), then drop the marker.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Test-only change: `Tests/UI/test_console_store_continuity.py::test_reattach_render_failure_retries_before_decisions_and_view_timers`.

- **Path.** After leaving for Library, the test monkeypatches `app._current_runtime_identity` to a server identity. `_reusable_navigation_screen` (`app_navigation.py`) then drops and disposes the cached Console instance, so the return builds a fresh `ChatScreen` (asserted: `reopened is not chat`) that re-attaches to the app-owned runtime. The authoritative `runtime_policy` state is unchanged: only the identity seam the route cache reads is flipped.
- **Intent kept.** A fresh view runs more syncs than the attach reconciliation (the old test's "first sync" was always the reconciliation's), and an injected raise in one of those failed its worker and exited the app. The injection now fires only while `_console_attach_reconcile_running`. Before the retry: no `remount_pending_approval` and no `_restore_collapsible_states` timer; after it: both, each recorded after the successful sync (`[True]`).
- **Settling.** The fresh view finishes reconciliation and first paint on its own worker after the sync returns; one `pilot.pause()` was not enough (three runs failed three different ways at load ~40: transcript "No messages yet.", transcript without the reply, `remount_after_sync == []`). It now waits (bounded, 10 s) for the remount and the rendered reply before asserting.
- **AC #2.** With the reconciliation's retry `set_timer` temporarily replaced by `return` in `chat_screen.py` (reverted): `TimeoutError` waiting for the retry (red), twice. Restored: green.
- **AC #3.** 3 of 3 passed (`-n 4`, under a temporary `@pytest.mark.bootstrap_profile`, removed before commit; plain, the file stops at `RecoveryRequired` locally, as TASK-39161 notes): 22.3 s, 16.5 s, 14.3 s call, load 42 / 38 / 39.
- No PR lane runs the file, and this task's criteria do not ask for one.
<!-- SECTION:NOTES:END -->
