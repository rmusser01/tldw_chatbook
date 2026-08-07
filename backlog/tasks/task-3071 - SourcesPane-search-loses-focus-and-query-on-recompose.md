---
id: TASK-3071
title: SourcesPane search loses focus and query on recompose
status: Done
assignee: []
created_date: '2026-08-07 16:17'
updated_date: '2026-08-07 17:01'
labels:
  - watchlists
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Sibling of the ItemsPane focus bug fixed in task-2513 Task 10: SourcesPane's recompose() override only preserves create-form fields, so a recompose while typing in the sources search box steals focus (and with select_on_focus=True, the refocus selects-all so the next keystroke replaces the half-typed query). Apply the same treatment: capture focused widget pre-teardown, restore caret-at-end via call_after_refresh, select_on_focus=False on the search input.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typing in sources search survives a pane recompose with focus and caret intact
- [x] #2 Half-typed query is not replaced on refocus
- [x] #3 Regression tests mirror the ItemsPane focus tests
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Mirror ItemsPane's task-2513 Task-10 fix: capture screen.focused pre-teardown in recompose(), restore caret-at-end via call_after_refresh
2. select_on_focus=False on the sources search input
3. Regression tests mirroring the ItemsPane focus tests
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Applied the ItemsPane task-2513 Task-10 treatment to SourcesPane, integrated with its existing create-form focus logic (TASK-1035/TASK-1345):

- `recompose()` now captures `self.screen.focused` pre-teardown (not `app.focused` -- ScreenStackError on transient empty stack, same reason as ItemsPane); when `#sources-search-input` held focus, it schedules `_restore_search_focus` via `call_after_refresh` and returns BEFORE the create-form path, so a still-armed `_pending_create_focus` can never yank the caret mid-keystroke (the TASK-1345 "current focus beats stale intent" ordering, extended to the search box).
- New `_restore_search_focus()` focuses the live replacement input, caret at end -- mirroring `ItemsPane._restore_search_focus`.
- `select_on_focus=False` on the search input (load-bearing: Textual's default True made the programmatic refocus select-all, so the next keystroke replaced the query).
- Docstring updated: two create-form cases become three focus cases.

Red-green verified: the new screen-level test (type "krebs" into the box, assert full value + has_focus on the live replacement input) FAILS on pre-fix dev and PASSES with the fix. Suites: Tests/Watchlists sources_pane + collections_screen 77/77, ruff clean.

Modified: tldw_chatbook/UI/Watchlists_Modules/sources_pane.py, Tests/Watchlists/test_watchlists_collections_screen.py
<!-- SECTION:NOTES:END -->
