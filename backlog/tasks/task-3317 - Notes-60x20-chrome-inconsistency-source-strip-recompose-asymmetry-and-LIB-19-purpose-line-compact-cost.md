---
id: TASK-3317
title: >-
  Notes 60x20 chrome inconsistency: source strip renders only after a full recompose, and the LIB-19 purpose line eats 4 of 10 compact list rows
status: To Do
assignee: []
created_date: '2026-08-09 09:45'
labels:
  - library
  - notes
  - ux
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while repairing `Tests/UI/test_library_shell.py` against the merged surface (task-3315). Two product observations at 60x20 that deserve an owner ruling rather than a silent test pin:

1. **Per-route chrome asymmetry (nondeterministic-looking UI).** The one-row Database|Files source strip (`#library-notes-source-strip`, file-notes workspace `b83852eda`) is composed by the screen only when a FULL screen recompose runs while the notes canvas is selected. Entering Notes by pressing the rail row goes through `_replace_library_browse_canvas` (PR #1439's fast path, `03cd682df`), which swaps the canvas in place — so the plain list view shows NO strip (shell 15 rows), while entering the editor/sync/loading views forces a full recompose and the strip appears (shell 14 rows). The same logical "Notes selected" state renders different screen chrome depending on which internal update path ran last. Task-3315 pinned this per-route truth in `_assert_task8_compact_chrome` (test file names this task); the product should either always render the strip for notes routes or never render it at compact.

2. **LIB-19 purpose line's compact cost.** `#library-notes-database-purpose` (task-2858, `a3591b503`) is styled as a "muted one-line treatment", but at width 60 it wraps to 3 rows + 1 margin — 4 of the compact Navigator's 10 list rows. The notes-adaptive 60x20 program (PR #1439) and the LIB-19 copy (PR #1420) merged a day apart and were never reconciled; a compact-mode treatment (e.g. hide or truncate the sentence when `_library_notes_compact`) would restore the reviewed 60x20 budgets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The source strip's presence for notes routes is deterministic across entry paths (always or never per terminal class), with the chosen direction recorded
- [ ] #2 The database-purpose sentence has an explicit compact-mode treatment (kept, truncated, or hidden) chosen by the owner, not by layout accident
- [ ] #3 `Tests/UI/test_library_shell.py`'s task8 pins are updated to the single ruled truth (removing the per-route `source_strip` fork task-3315 had to pin)
<!-- AC:END -->

## Update — AC#1 closed dev-side (2026-08-09)

Observation 1 was fixed independently on dev by `d1df7d0a7` (TASK-13213,
"restore file notes source access"), which lands the fix in the direction this
task asked for: `_replace_library_browse_canvas` now refuses its targeted swap
whenever the mounted contextual chrome disagrees with the destination
(`notes_source_strip_mounted != (shell.canvas_kind == "notes")`), so entering
Notes by rail press goes through the full recompose and shows the strip exactly
like the editor/sync/loading routes. Every Database-notes route now settles at
3 + 1 + 1 + 14 + 1 at 60x20.

Confirmed empirically while rebasing the media-ingest follow-up branch onto dev
`f6911b37b`: the 60x20 `normal` case and the navigator `compact_surplus` case
both fail their old strip-less pins against a `git archive` extraction of dev's
product tree, i.e. the change is dev's, not the arc's. The pins were re-measured
in task-3315's round-2 addendum.

AC#2 is untouched (the LIB-19 sentence still costs 4 of the compact list's rows;
at 60x20 the navigator list is now 5). AC#3 is only partly satisfied: the
per-route `source_strip` fork survives for the create-note canvas
(`canvas_kind "notes-create"`), which still never composes the strip.
