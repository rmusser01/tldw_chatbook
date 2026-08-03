---
id: TASK-2066
title: >-
  MCP: footer shortcut bar stops advertising keys that do not work in context
  (F-055)
status: To Do
assignee: []
created_date: '2026-08-03 17:24'
labels:
  - ux-review
  - mcp
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Footer shows 'space cycle permission' in all four modes but the key only works in Permissions with the matrix focused; pressing t in Servers mode force-switches to Tools and notifies 'Select a tool first.' Evidence: mcp_screen.py:30-41. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Footer shortcut context is per-mode and only shows working keys,t with no tool selected is a no-op with a hint (no mode hijack),Tests updated
<!-- AC:END -->
