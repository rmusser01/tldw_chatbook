---
id: TASK-32272
title: >-
  Library Notes select mode shows two selection counters that disagree: 1
  selected above 0 selected
status: Done
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 15:19'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed by the peer session at 235x52 on a seeded profile at dev `3315241674`, and not filed anywhere: in Notes select mode the toolbar reads "1 selected" while the line directly below it reads "0 selected". Two counters for one selection, on the same pane, at the same moment, disagreeing -- and the count is the only feedback the mode gives about what a bulk action will touch.

Cause INFERRED, not traced: two independently maintained counts, one updated on the row toggle and one on the selection-set change.

Related but distinct: task-32261 covers the same strip printing `0 selected` twice and losing "Export selected" at 100x30. This task is the two counters holding **different** numbers.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every visible selection count on the Notes canvas reads the same number at all times
- [x] #2 There is one source of truth for the count, pinned by a test that toggles a single row and asserts both labels
- [x] #3 Verified live at 235x52 with a capture
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live at 235x52: enter select mode, toggle one row, read both counts.
2. Trace: compose reads list_state.selected_count for both, the in-place toggle patcher updates only the toolbar one.
3. RED test in test_library_multiselect_notes.py toggling one row and asserting both labels.
4. Fix _apply_library_row_toggle to write one label string to every count the kind renders.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause, proven.** Reproduced live at 235x52 on a seeded profile
(`wave3-caps/list-tree/12-select-counters-before.txt`): the toolbar reads
"1 selected" and the line below reads "0 selected". The Notes canvas
composes the count twice — `#library-notes-selected-count` in the toolbar
and `#library-notes-selection-status` under it — and both read the same
`list_state.selected_count` at compose time. But a row press does not
recompose: it goes through `_apply_library_row_toggle`
(`UI/Library_Modules/canvas_sync.py`), which only ever patched the
toolbar one. So the two agree until the first toggle and never again.
Not "two independently maintained counts" as the report inferred — one
count, one stale renderer.

**Fix.** The patcher builds the label once and writes it to every count
the kind renders; the id query is empty for media and conversations,
which have only the toolbar counter, so they are untouched.

**Coverage.** `test_both_notes_selection_counts_move_together_on_one_toggle`
toggles a single row and asserts both labels, in both directions (RED:
`['1 selected', '0 selected']`). Verified live at 235x52
(`21-select-counters-after-wide.txt`) and 100x30
(`24-select-counters-after-compact.txt`).

**Files.** `tldw_chatbook/UI/Library_Modules/canvas_sync.py`,
`Tests/UI/test_library_multiselect_notes.py`,
`Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
