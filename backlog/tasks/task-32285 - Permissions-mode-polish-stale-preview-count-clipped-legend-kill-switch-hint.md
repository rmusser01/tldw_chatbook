---
id: TASK-32285
title: 'Permissions mode polish: stale preview count, clipped legend, kill-switch hint'
status: To Do
assignee: []
created_date: '2026-09-10 19:15'
labels:
  - mcp
  - permissions
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After Space set fs_read to Allow the preview still read '0 allow, 30 ask, 0 off'; the legend and gate breadcrumb clip mid-sentence at 50 rows; the kill-switch hint names only calculator and date/time; the kill switch has two differently worded refusal strings. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The preview counts update on every state change.
- [ ] #2 The legend and gate breadcrumb are fully readable at 50 rows.
- [ ] #3 The kill-switch hint describes its real blast radius and one refusal wording is used everywhere.
<!-- AC:END -->
