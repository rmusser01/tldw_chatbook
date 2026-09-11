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
- [ ] #1 A documented screen binding shown in the footer legend moves focus to the card's first undecided decision.
- [ ] #2 The chat tab's needs-approval marker also jumps to the card.
- [ ] #3 The route works at 80 columns with the inspector closed, and the card is reachable in the Tab order from the composer within a small tested number of presses.
<!-- AC:END -->
