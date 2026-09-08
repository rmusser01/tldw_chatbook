---
id: TASK-32100
title: >-
  Library Notes: opening a note supersedes the folder-tree locator, so 'Locating
  note…' / row reveal is abandoned on every open
status: To Do
assignee: []
created_date: '2026-09-08 22:42'
labels:
  - library
  - notes
  - ux
  - critique-8
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After task-32050 the detail load no longer depends on the tree tokens, but the row click still bumps the notes navigation generation and cancels `_locate_library_notes_tree_target`, so the tree never reveals the opened note's row. Found by the task-32050 review (PR #2519). Rider from the critique-8 fix wave reviews (plan Docs/superpowers/plans/2026-09-08-library-crit8-wave.md; wave PRs #2519 #2523 #2524 #2525 #2528 #2531).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening a note from the list reveals and marks its row in the folder tree
- [ ] #2 A second click while the locator runs supersedes only the older locator, not the detail load
<!-- AC:END -->
