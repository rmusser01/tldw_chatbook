---
id: TASK-32179
title: >-
  Library Notes: LibraryScreen._seed_local_source_snapshot_from_cache is
  defined twice
status: To Do
assignee: []
created_date: '2026-09-09 09:17'
updated_date: '2026-09-09 09:17'
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
- [ ] #1 `LibraryScreen` defines `_seed_local_source_snapshot_from_cache`
  exactly once
- [ ] #2 A test asserts that the surviving definition is the one callers
  actually reach (guarding against the duplicate silently reappearing)
<!-- AC:END -->
