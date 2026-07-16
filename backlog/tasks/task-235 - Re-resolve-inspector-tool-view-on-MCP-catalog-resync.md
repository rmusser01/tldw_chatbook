---
id: TASK-235
title: Re-resolve inspector tool view on MCP catalog resync
status: To Do
assignee: []
created_date: '2026-07-16 15:19'
labels:
  - mcp
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The inspector's tool detail (schema, stale flag) is captured at selection time and never re-resolved when a refresh/lifecycle action changes the catalog — detail can contradict the table until reselection. Run against a vanished tool fails cleanly, so this is staleness, not a crash (PR #639 final review N2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Catalog resync re-resolves the currently shown tool by identity,Vanished tool clears the inspector tool view,Changed schema is reflected in an open Test panel or the panel is closed with a notify
<!-- AC:END -->
