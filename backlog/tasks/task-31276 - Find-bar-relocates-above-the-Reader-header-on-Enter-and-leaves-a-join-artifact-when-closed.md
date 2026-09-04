---
id: TASK-31276
title: Find bar relocates above the Reader header on Enter and leaves a join artifact when closed
status: To Do
assignee: []
created_date: '2026-09-04 13:54'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 P2: pressing Enter in the Find bar moved the whole bar from under the `Read` label to the top of the pane above `Local Media item`, pushing the header down six rows (B cap_20). After Escape closes Find, a five-cell `┐─────Local Media item` artifact appears at the pane join and persists across later interactions (14 captures; absent on a fresh open).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Find bar stays in one place (under the mode row) through open, typing, Enter and match navigation
- [ ] #2 No `┐─────` artifact at the pane join after Find close, tab clicks or the More menu
- [ ] #3 Live-verified at 235x52 and 100x30 with captures in the notes
<!-- AC:END -->
