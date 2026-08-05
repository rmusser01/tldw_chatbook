---
id: TASK-723
title: Decide how Default-workspace chats are classified in the conversation browser
status: Done
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
- [x] #1 A documented decision states how Default-workspace conversations are grouped and why
- [x] #2 Browser grouping and switcher copy agree with that decision
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Decide, record as ADR, align switcher copy with the browser grouping, lock with a test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Decision recorded as backlog/decisions/027-default-workspace-chats-in-chats-section.md: Default-workspace conversations stay in the Chats section (everyday chatting must not demand workspace vocabulary; a perpetual Default group would kill the honest "No workspace conversations." empty state; storage identity stays an implementation detail). Alignment: the switcher's Default row is annotated "Default (everyday chats)" so switcher copy and browser grouping tell one story; Default is rename/archive-protected (TASK-714) so the anchor stays stable. Test: test_default_row_labeled_everyday_chats in Tests/UI/test_console_workspace_lifecycle.py.
<!-- SECTION:NOTES:END -->
