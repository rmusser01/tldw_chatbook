---
id: TASK-32453
title: 'MCP Hub UX Wave B: safe permission cycle order'
status: In Progress
assignee: []
created_date: '2026-09-11 19:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reorder the Permissions Space-cycle so the first press from Inherit lands on Ask, not Allow (Inherit → Ask → Allow → Off). Allow becomes a deliberate second press. Store helper, legend copy, and every pinned test move together in one change (UX program 2026-09-11, safety assumption adopted at review).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 cycle_ui_state(None) returns ask (not allow),Legend reads Space cycles Inherit → Ask → Allow → Off,All pinned tests updated to the new order in the same change,Permission regression suites green
<!-- AC:END -->
