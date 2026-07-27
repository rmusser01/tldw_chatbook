---
id: TASK-897
title: >-
  The Sources toolbar starves its own DataTable down to a single visible row
status: To Do
assignee: []
created_date: '2026-07-27 16:00'
labels:
  - watchlists
  - bug
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In the real Watchlists shell, `SourcesPane`'s `#sources-toolbar` claims roughly 33 of the 34 rows available to the pane, leaving its `DataTable` **one visible row** — regardless of terminal size. Making the terminal bigger does not help, because the toolbar takes whatever is available rather than what it needs.

So the Sources section, which is the screen's main list of what a user is monitoring, shows one source at a time.

Found while implementing task-876 (tree and pane selection affordances), measured against the production stylesheet in the full shell rather than a bare harness. It is pre-existing and unrelated to that task's changes, so it was reported rather than folded in.

This is the same class of defect as the FEEDS region clipping its own content, fixed in Phase C: a height rule that is correct in isolation and wrong once the widget is nested in the real layout. That one was invisible to every task-level test because they all used a bare `App` with no stylesheet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Sources DataTable shows as many rows as the pane's height allows, not one
- [ ] #2 The toolbar takes only the height its own content needs
- [ ] #3 Verified at both 160x42 and the app's real 235x52, against the production stylesheet in the full shell — not a bare `App` harness
- [ ] #4 A test fails if the table's visible row count collapses again, and is proven to fail against the current code before the fix
- [ ] #5 The other section panes are checked for the same toolbar-vs-table height pattern, and any found are listed here or fixed
<!-- AC:END -->
