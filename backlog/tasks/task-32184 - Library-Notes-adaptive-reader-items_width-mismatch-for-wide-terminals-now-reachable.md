---
id: TASK-32184
title: >-
  Library Notes: adaptive reader items_width mismatch for wide terminals now
  reachable
status: Done
assignee: []
created_date: '2026-09-09 17:53'
updated_date: '2026-09-11 10:45'
labels:
  - library
  - notes
  - tests
  - follow-up
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Unblocking task-32175's row-0 wait exposed a downstream, unrelated pre-existing defect in test_library_shell.py::test_library_production_width_matrix_custom_preferences: for terminal widths 120/170/235 (all four saved_width values), the resolved neutral layout's items_width does not match the expected 56 once a Database Notes row is opened and the adaptive reader layout is re-synced from the shell. The mismatch is not one constant: the 235- and 170-column cases compute 64 (`assert 64 == 56`), the three 120-column cases compute 58 (`assert 58 == 56`) — so whoever picks this up should not chase a single wrong number. The test previously died earlier on the flat #library-notes-row-0 wait, so this assertion was never reached before. No production code was touched by task-32175 (test-only); this is a pre-existing geometry bug in the adaptive reader layout sync path, independent of the flat-list vs folder-tree row composition.

Reproduction: run test_library_shell.py::test_library_production_width_matrix_custom_preferences after task-32175's fix lands — 12 of 24 parametrized cases fail with `AssertionError: assert neutral.items_width == neutral_items_width` at the assert following `screen._sync_library_notes_reader_layout_from_shell()`. Narrower widths (60/80/100) pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The 12 currently-red parametrized cases (terminal_width in 120/170/235, all four saved_width values) of test_library_production_width_matrix_custom_preferences pass
- [x] #2 A Database Notes reader opened at 120, 170 and 235 columns resolves the same items_width through the shell sync as `resolve_adaptive_reader_layout` resolves standalone, so the two agree at every width rather than only the narrow ones
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure what the shell sync and a standalone resolve actually produce at
   every width/saved-width pair, UNDER pytest (the config differs from a bare
   script -- see the notes).
2. Re-pin the matrix to that truth and add the agreement assertion AC#2 asks for.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The two paths already AGREE -- AC#2's premise is wrong. Measured through a
temporary pytest probe that mirrors the test body at all 24 pairs: the shell
sync and `resolve_adaptive_reader_layout` over the same shell width return the
same `items_width` at every one (235/170 -> 64, 120 -> 58, 100 -> 0, 80/60 ->
0). The mismatch was entirely in the test's constants.

Cause of the stale constants: task-32127 (2026-09-08, "give the Notes list the
width and a truthful toolbar") gave `LIBRARY_NOTES_READER_PROFILE`
`list_grows=True` and a 64-cell `list_comfort_width`, and resolved the layout
with `reader_has_item`. So the list now takes the surplus up to 64 (235/170),
and below that ceiling everything the work pane's 48-cell floor and the two
5-cell grips leave (116-10-48 = 58). The 100-column row was stale for a second
reason: the saved Items width is `ITEMS_TARGET_WIDTH` (50), and 10+50+48 = 108
does not fit 100, so the pane collapses -- that row was red on dev too (dev
ff2dc03145: 16 of 24 red, not the 12 this task recorded at filing time).
`priority_items_width` was likewise re-pinned from 40 to `ITEMS_TARGET_WIDTH`.

Trap worth knowing (cost a wrong diagnosis): the same probe run as a BARE
script reads the developer's real `~/.config/tldw_cli/config.toml` and saw
`items_width = 40`, which reproduced none of the failures. Under pytest the
config is isolated and the default 50 applies. Measure geometry under pytest.

AC#2 is now a live pin: after the neutral block the test asserts
`neutral.items_width == resolve_adaptive_reader_layout(shell width, work-first
prefs, profile, reader_has_item=...).items_width`, so the next profile change
fails as a disagreement rather than as a constant.

RED: 12 of 24 parametrisations (`assert 64 == 56`, `assert 58 == 56`), 16 on
dev. GREEN: 24 passed.

Task-8 review (finding 9) doubted the 100-column arithmetic and proposed that
the row collapses because the work-first prefs carry `library_open=True`.
Re-measured under pytest with `resolve_adaptive_reader_layout(100, prefs,
LIBRARY_NOTES_READER_PROFILE, reader_has_item=True)`:

| custom_widths_enabled | library_open | items_width pref | items pane |
|---|---|---|---|
| True | True | 50 | closed, 0 (reader 90) |
| True | False | 50 | closed, 0 (reader 90) |
| True | either | 42 | open, 42 (reader 48) |
| False | either | 50 or 42 | open, 42 (reader 48) |

So the library flag makes no difference (the library pane closes at 100
columns either way) and the saved 50-cell Items width IS what closes the
pane: a custom width is honoured exactly or not at all, and 10 + 50 + 48 does
not fit 100 while 42 does. The review's `items_width=42, items_open=True`
reading is the custom-widths-OFF row of this table, not the test's
configuration. The explanation above stands as written.

Files: `Tests/UI/test_library_shell.py`.
<!-- SECTION:NOTES:END -->
