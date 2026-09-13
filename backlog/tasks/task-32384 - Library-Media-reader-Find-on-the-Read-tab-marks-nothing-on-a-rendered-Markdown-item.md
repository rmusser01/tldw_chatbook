---
id: TASK-32384
title: 'Library Media reader: Find on the Read tab marks nothing on a rendered Markdown item'
status: To Do
assignee: []
created_date: '2026-09-11 10:30'
labels:
  - library
  - media
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32365's fix makes a Find query drop a rendered Markdown *analysis* to the raw view that actually marks matches, by branching `sync_query_state` on `forces_analysis_raw`. That gate is `reader_mode == "analysis"`, so the Read tab keeps the original patch-in-place branch and has the identical shape: a Markdown transcript opens rendered, a submitted query reports matches, and nothing is highlighted. The Task 1 report named this explicitly as an open rider rather than half-fixing it -- a patch cannot change which widget is mounted, so the Read tab needs the same recompose seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Submitting a Find query over a rendered Markdown item on the Read tab shows the view where matches are marked
- [ ] #2 Clearing the query restores the rendered view, and the Rendered control is refused with its reason while a query is active
- [ ] #3 The Read tab's behaviour is pinned by its own test, alongside the existing Analysis-tab pin
<!-- AC:END -->
