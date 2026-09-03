---
id: TASK-28008
title: Library media list - show analysis presence on rows
status: To Do
assignee: []
created_date: '2026-09-02 04:10'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The browse summary projection is contractually exactly five keys (library_media_state.py:52-54 validator), so a row cannot show whether the item has an analysis - the product's core artifact is invisible at every list-level decision point. Extend the projection with has_analysis, bump the key-set contract, and render a one-glyph marker in the row secondary line and the preview pane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Rows visibly distinguish items with an analysis from items without
- [ ] #2 The summary key-set contract and its tests are updated
- [ ] #3 No per-row extra DB round-trips (presence comes from the existing summary query)
<!-- AC:END -->
