---
id: TASK-28015
title: Library media Trash - Restore action renders detached far below the item row
status: To Do
assignee: []
created_date: '2026-09-02 04:11'
labels:
  - library
  - bug
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live-observed: the Trash view's only action, Restore, renders at the very bottom of the canvas with roughly 40 blank rows between the item rows and the button; clicking a row or pressing Enter reveals no per-item actions. Permanent delete / Empty Trash is already tracked as task-15130 - this task covers only the layout and association defect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Restore is visually associated with the trashed item list (no dead gap)
- [ ] #2 A keyboard path from a trash row to Restore exists
<!-- AC:END -->
