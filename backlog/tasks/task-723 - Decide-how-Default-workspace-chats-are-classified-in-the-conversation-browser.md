---
id: TASK-723
title: Decide how Default-workspace chats are classified in the conversation browser
status: To Do
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - ux
  - console
  - workspaces
  - design
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Default-workspace conversations carry workspace_id=workspace-default and scope_type=workspace in the DB but the browser files them under Chats while other workspaces get named groups (cap-08/29 + DB check). This contradicts the switcher's framing of Default as a workspace like the others; users adopting workspaces later will look for old chats in the wrong bucket. Decide and make the mental model consistent (either Default is a visible group, or copy explains that Default chats live in Chats). Finding m3.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A documented decision states how Default-workspace conversations are grouped and why
- [ ] #2 Browser grouping and switcher copy agree with that decision
<!-- AC:END -->
