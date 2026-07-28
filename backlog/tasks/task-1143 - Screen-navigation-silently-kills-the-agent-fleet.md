---
id: TASK-1143
title: Screen navigation silently kills the agent fleet
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 18:05'
updated_date: '2026-07-28 04:00'
labels:
  - console
  - ux
  - agents
  - uat
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT (Docs/superpowers/qa/parallel-agents-uat-2026-07-27, F5): navigating away from Console (e.g. to Settings to change the run cap) unmounts the screen, shuts down the controller, and denies every in-flight/parked run — by design (instance lifecycle) — but nothing warns before, and nothing reports after: returning shows a fresh Console with no markers, toasts, or record of the killed runs. Users running parallel background agents lose them by opening Settings. Add a confirm-on-navigate when the fleet is busy, and/or a returning notice ("N runs were cancelled when you left Console"), and document the lifecycle in the user guide.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Navigating away with in-flight or parked runs either asks for confirmation or leaves a visible record on return.
- [x] #2 Never auto-approves; deny-on-teardown semantics unchanged.
- [x] #3 User guide documents that runs are Console-screen-scoped.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Investigate the navigation seam: app.py's handle_screen_navigation already
   consults an outgoing screen's flush_pending_work() (await, False vetoes)
   before switch_screen -- confirmed this pattern is duck-typed and reusable;
   add a sibling confirm_navigation() hook consulted the same way.
2. Add ConsoleChatController.busy_fleet_session_count(): union of the
   existing in_flight_run_count() live-busy-session set and every live
   session with an outstanding approval-like round (_pending_approvals,
   covers MCP/skill-install/skill-script rounds via add_pending_round) --
   no new "busy" definition, just the union of the two existing predicates.
3. ChatScreen.confirm_navigation(): idle fleet returns True immediately (no
   dialog); busy fleet shows the existing ConfirmationDialog
   ("N agent runs will be cancelled if you leave Console. Leave anyway?",
   Leave/Stay) and returns the user's choice. push_screen_wait requires a
   worker context (NoActiveWorker otherwise), so the wait is delegated to
   self.run_worker(...) and awaited back via worker.wait().
4. ChatScreen.on_unmount: snapshot busy_fleet_session_count() BEFORE calling
   shutdown() (unchanged) and, if non-zero, stash it on the App
   (TldwCli._console_fleet_teardown_notice) -- the app outlives the screen.
5. ChatScreen.on_mount: consume+clear that slot and show one toast
   ("N agent runs were cancelled when you left Console.") when non-zero;
   silent otherwise.
6. Document the lifecycle + both guards in Docs/User_Guide/index.md (no
   dedicated Console page exists yet).
7. TDD: real-navigation-seam tests (NavigateToScreen -> handle_screen_
   navigation) for confirm/veto wiring, idle-fleet silence, and the
   record-after toast; run the three required gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented BOTH halves (confirm-before + record-after), reusing existing
predicates and dialog patterns end to end.

Approach:
- ConsoleChatController.busy_fleet_session_count() (Chat/console_chat_
  controller.py): union of the existing in_flight_run_count() live-busy
  set and every LIVE session with an outstanding approval-like round
  (_pending_approvals -- covers MCP tool approvals, skill-install, and
  skill-script confirms, all registered via add_pending_round). No new
  "busy" definition; it's the union of two predicates the fleet UX
  already relied on.
- app.py's handle_screen_navigation already awaits an outgoing screen's
  flush_pending_work() (False vetoes the switch) before switching --
  added a sibling confirm_navigation() hook consulted the same way,
  fail-closed on exception, right after the flush check.
- ChatScreen.confirm_navigation(): idle fleet returns True immediately
  (no dialog, no delay -- satisfies "single-session users who never run
  agents see zero new prompts"). Busy fleet shows the existing
  ConfirmationDialog ("N agent runs will be cancelled if you leave
  Console. Leave anyway?", Leave/Stay). push_screen_wait requires a
  worker context (raises NoActiveWorker from an ordinary message
  handler, which handle_screen_navigation is) -- delegated the wait to
  self.run_worker(..., exit_on_error=False) and awaited the result via
  worker.wait().
- ChatScreen.on_unmount snapshots busy_fleet_session_count() BEFORE
  calling the existing (unchanged) controller.shutdown(), and if
  non-zero stashes it on TldwCli._console_fleet_teardown_notice -- a
  small App-level attribute, since the App outlives the doomed screen
  and screens are never cached (_create_navigation_screen always builds
  fresh instances).
- ChatScreen.on_mount calls _notify_console_fleet_teardown_if_any(),
  which consumes+clears that slot and shows one toast ("N agent runs
  were cancelled when you left Console.") when non-zero; silent
  (0 = never set) on an ordinary mount or a second consecutive mount.
- Documented the Console-screen-scoped lifecycle and both guards in
  Docs/User_Guide/index.md (no dedicated Console page exists yet per
  the User Guide program's G1 status -- added a short section to the
  index instead of stubbing a new page out of scope).

Deny-on-teardown semantics (AC#2) are untouched: shutdown() itself was
not modified, so nothing is ever auto-approved by this change.

Testing (TDD, real navigation seam):
- Tests/UI/test_screen_navigation.py: two new unit-level tests mirroring
  the existing flush-veto tests exactly --
  test_navigation_confirms_with_outgoing_screen_and_honors_veto (False
  vetoes, True proceeds) and
  test_navigation_confirm_exception_warns_and_aborts_switch (fails
  closed).
- Tests/UI/test_console_parallel_runs.py:
  test_navigating_away_with_busy_fleet_confirms_and_records_teardown --
  one continuous journey on the REAL running TldwCli app (NavigateToScreen
  posted, handle_screen_navigation dispatched for real, not a synthetic
  call) covering: idle fleet -> instant navigation, no dialog/toast; busy
  fleet -> dialog shown, Stay aborts (screen + run survive), Leave
  proceeds (run torn down, count recorded); next Console mount -> toast
  once with the right N, second mount silent.
- Harness note: while the confirm dialog is open, TldwCli's own message
  pump is legitimately suspended inside handle_screen_navigation (awaiting
  the user's choice), so Pilot.pause()'s internal idle-wait (which needs
  that same pump to process one more message) hangs for its full 30s
  timeout if called during that window. The real-navigation journey test
  polls app.screen via plain asyncio.sleep instead of pilot.pause() around
  those windows, and polls for on_unmount's _console_chat_controller ->
  None / the toast slot clearing rather than racing switch_screen's
  synchronous screen-stack update (which flips app.screen before the
  outgoing/incoming screen's on_unmount/on_mount actually runs).

Gates run (foreground, worktree .venv):
- Tests/UI/test_console_parallel_runs.py + Tests/UI/test_screen_
  navigation.py: 84 passed.
- Tests/UI/test_console_mcp_approval.py: 42 passed, 2 known pre-existing
  failures (CSS-geometry zero-size assertion, MCP execution-log
  cancellation-reason string) -- unrelated to this change.

Modified/added files:
- tldw_chatbook/app.py (confirm_navigation consult + App-level notice slot)
- tldw_chatbook/Chat/console_chat_controller.py (busy_fleet_session_count)
- tldw_chatbook/UI/Screens/chat_screen.py (confirm_navigation,
  _notify_console_fleet_teardown_if_any, on_mount/on_unmount wiring)
- Docs/User_Guide/index.md (new "Console agent runs are screen-scoped"
  section)
- Tests/UI/test_screen_navigation.py, Tests/UI/test_console_parallel_runs.py
<!-- SECTION:NOTES:END -->
