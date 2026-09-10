---
id: TASK-32179
title: >-
  Library Notes: LibraryScreen._seed_local_source_snapshot_from_cache is defined
  twice
status: Done
assignee: []
created_date: '2026-09-09 09:17'
updated_date: '2026-09-09 17:34'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - library-screen
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from the Notes critique fix wave (plan
Docs/superpowers/plans/2026-09-09-library-notes-critique-wave.md); raised in
the final whole-branch review of the wave. `LibraryScreen._seed_local_source_snapshot_from_cache`
is defined twice in `library_screen.py` (around line 9353 and again around
line 11729); this predates the wave, but no earlier task caught it. Python
method resolution silently keeps only the second definition, so the first is
dead code that a future edit could waste time on.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `LibraryScreen` defines `_seed_local_source_snapshot_from_cache`
  exactly once
- [x] #2 A test asserts that the surviving definition is the one callers
  actually reach (guarding against the duplicate silently reappearing)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Delete the dead first _seed_local_source_snapshot_from_cache definition in library_screen.py (the no-arg, None-returning stub at the old ~9353) -- Python method resolution already kept only the second, now-real definition (accepts now, returns bool) for both call sites.
2. Add an AST-based test asserting exactly one definition of that name in LibraryScreen's class body, plus a signature check confirming the surviving definition is the real cache-applying implementation (accepts now), not the dead stub -- guards against the duplicate silently reappearing.
3. Run the new test RED (2 definitions found) then GREEN after the deletion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Deleted the dead first _seed_local_source_snapshot_from_cache definition in library_screen.py (the no-arg, None-returning stub, originally ~line 9353) -- Python method resolution already made only the SECOND definition (accepts `now`, returns bool, clones the snapshot, tracks the reconcile generation) reachable from both call sites (__init__ and restore_state); the first was pure dead weight.

Added test_seed_local_source_snapshot_from_cache_is_defined_exactly_once (Tests/UI/test_library_notes_riders_r_editor.py) as an AST-based guard: parses LibraryScreen's class body and asserts exactly one FunctionDef with this name, independent of Python's own runtime shadowing, plus a signature check that the surviving definition is the real cache-applying implementation (has a `now` parameter) rather than the old stub -- catching a reintroduced duplicate even if it happened to come first.

RED before the fix (2 definitions found, AssertionError); GREEN after (1 definition, real implementation confirmed).

Files: tldw_chatbook/UI/Screens/library_screen.py (deletion only), Tests/UI/test_library_notes_riders_r_editor.py (new test appended to the shared r-editor riders file already created by task-32177's commit).
<!-- SECTION:NOTES:END -->
