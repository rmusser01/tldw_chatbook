---
id: TASK-2087
title: 'Roleplay: give Inspector Conversations an empty state (F-036)'
status: To Do
assignee: []
created_date: '2026-08-03 17:24'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
show_conversations(()) is called without empty_copy, leaving a dangling 'Conversations' header pre-selection. Evidence: personas_inspector_pane.py:198-209. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Conversations section shows empty copy or is hidden when empty,Tests updated
<!-- AC:END -->
