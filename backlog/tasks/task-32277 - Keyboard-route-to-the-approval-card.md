---
id: TASK-32277
title: Keyboard route to the approval card
status: Done
assignee: []
created_date: '2026-09-10 19:11'
updated_date: '2026-09-11 00:34'
labels:
  - console
  - approvals
  - keyboard
  - accessibility
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the composer, Tab never reaches the approval card (twelve presses cycle the header action bar). The only entry points are the Approvals chip, which scrolls off the status strip below roughly 204 columns, and the inspector's 'Review approval' button, which sits below the fold at 50 rows. The card has no key bindings. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A documented screen binding shown in the footer legend moves focus to the card's first undecided decision.
- [x] #2 The chat tab's needs-approval marker also jumps to the card.
- [x] #3 The route works at 80 columns with the inspector closed, and the card is reachable from the composer with one keypress (Alt+A).
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the existing "Review approval" inspector-button code path
   (`ChatScreen.handle_console_inspector_review_approval`) and the
   session-tab press handler (`console-session-tab-` branch in
   `on_button_pressed`, delegating to `ConsoleSessionController.
   _handle_console_session_tab_press`).
2. Write failing tests for: the `alt+a` `Binding` exists with `show=True`
   and an implemented action; with a pending batch, the action leaves
   focus on `.approval-row-decision`; with nothing pending, it notifies
   `CONSOLE_INSPECTOR_NO_APPROVAL_REASON`; clicking a session tab wearing
   the `◆` (`ConsoleRunMarker.NEEDS_APPROVAL`) marker reaches the same
   route; the whole thing still works at 80 columns with the inspector
   closed.
3. Extract the inspector button's body into a shared
   `ChatScreen._route_console_pending_approval_focus()`, call it from the
   button handler, a new `action_review_pending_approval()` (the `alt+a`
   binding's target), and a new branch in the `console-session-tab-`
   press dispatch that activates a backgrounded NEEDS_APPROVAL session
   first (un-parking its round) before routing.
4. Add `("Alt+A", "approval")` to `CONSOLE_WORKBENCH_SHORTCUTS` next to
   `("Alt+I", "inspect")`, updating the one pinned footer-string test.
5. Run the new/updated tests plus `test_console_agent_steering_bar.py`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a screen-level alt+a binding (action_review_pending_approval) that routes to the pending MCP approval card, extracting the inspector Review-approval button's body into a shared ChatScreen._route_console_pending_approval_focus() so the button, the new binding, and a click on a session tab wearing the NEEDS_APPROVAL (diamond) marker all share one implementation; a marker click on a backgrounded session activates it first (un-parking its round via the existing park/switch_session machinery) before focusing the card's first undecided .approval-row-decision Select, never Submit. Advertised Alt+A approval in CONSOLE_WORKBENCH_SHORTCUTS next to Alt+I inspect, updating the one pinned footer-string test. TDD: 6 new/updated tests across Tests/UI/test_console_mcp_approval.py (binding registration, focus-on-pending, notify-when-empty, 80-column route with inspector closed), Tests/UI/test_console_parallel_runs.py (tab-marker click on a parked background session), and Tests/UI/test_console_workbench_contract.py (footer legend pin) -- verified RED against the unmodified production code via a scoped git stash of chat_screen.py, then GREEN after restoring it; a full-file rerun of all four target test files showed zero new failures versus a pre-change baseline (20 pre-existing flaky failures at baseline, 18 of the same names after, none new). Review fix round: reordering the literal DOM Tab/Shift+Tab focus chain so the card sits in sequential tab order was ruled out of scope -- Alt+A's one-keypress route is what AC#3 now describes -- and Alt+A was added to CONSOLE_WORKBENCH_SHORTCUT_GROUPS's "Panes" group ("Review pending approval") next to Alt+I so the never-truncating F1 help panel, not just the footer, carries the route (mirrors the existing TASK-24704 Alt+I precedent and its pinning test).
<!-- SECTION:NOTES:END -->
