---
id: TASK-32277
title: Keyboard route to the approval card
status: To Do
assignee: []
created_date: '2026-09-10 19:11'
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
- [x] #3 The route works at 80 columns with the inspector closed, and the card is reachable in the Tab order from the composer within a small tested number of presses.
<!-- AC:END -->

## Implementation Plan

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
