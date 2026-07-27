---
id: TASK-897
title: >-
  The Sources toolbar starves its own DataTable down to a single visible row
status: Done
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
- [x] #1 The Sources DataTable shows as many rows as the pane's height allows, not one
- [x] #2 The toolbar takes only the height its own content needs
- [x] #3 Verified at both 160x42 and the app's real 235x52, against the production stylesheet in the full shell — not a bare `App` harness
- [x] #4 A test fails if the table's visible row count collapses again, and is proven to fail against the current code before the fix
- [x] #5 The other section panes are checked for the same toolbar-vs-table height pattern, and any found are listed here or fixed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`#sources-toolbar` is a bare `Vertical` wrapping several `.destination-filter-strip` rows and had no height rule anywhere in the stylesheet, so it inherited Textual's `height: 1fr` default. Its size therefore came from whatever `#sources-table` was not using, rather than from its own controls. Fixed with `#sources-toolbar { height: auto; }` in `features/_watchlists.tcss`.

**The reported symptom was real but the stated trigger was wrong, and finding that mattered.** The report said the toolbar claims ~33 of 34 rows regardless of terminal size. Measured against current dev with 12 sources in the table: toolbar 3, table 13 — no defect. It only appears when the table is **empty**: toolbar 15 of a 16-row pane at 160x42, and 25 of 26 at 235x52, with the table collapsed to a single row.

That is the worse case, not a milder one. An empty Sources section is the first thing a new user sees, and it is exactly when the pane should be showing its empty state rather than a ballooned filter bar. A test written against the reported trigger passed against the unfixed code — the first version of this test did, which is how the discrepancy surfaced.

**AC #5 sweep.** The other four toolbars (`runs`/`items`/`rules`/`notifications`) are `Horizontal` with `.destination-filter-strip`, which pins `height: 1` in `layout/_panes.tcss`. They were never affected. `#sources-toolbar` is the only wrapper of its kind in the package and the only one needing its own rule.

Test is parametrized over 160x42 and 235x52 in the full shell under the production stylesheet, and asserts on the empty state before populating rows. Proven red at both viewports before the fix.
<!-- SECTION:NOTES:END -->
