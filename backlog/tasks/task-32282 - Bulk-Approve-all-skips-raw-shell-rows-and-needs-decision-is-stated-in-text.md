---
id: TASK-32282
title: Bulk Approve all skips raw-shell rows and needs-decision is stated in text
status: To Do
assignee: []
created_date: '2026-09-10 19:13'
labels:
  - console
  - approvals
  - accessibility
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
'Approve all' moves a raw-shell row off its deliberate Deny default, and the needs-decision text prefix is never produced (no producer sets it; the bulk-skip path only adds a CSS class), so the state is colour-only. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Approve all leaves raw-shell rows on Deny and flags them as needing an explicit decision.
- [ ] #2 A row skipped by a bulk action shows a 'needs decision' prefix in its header text until it is decided.
- [ ] #3 Tests cover both behaviours.
<!-- AC:END -->
