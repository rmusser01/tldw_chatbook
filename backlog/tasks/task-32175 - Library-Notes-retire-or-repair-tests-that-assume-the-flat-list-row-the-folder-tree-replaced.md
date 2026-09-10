---
id: TASK-32175
title: >-
  Library Notes: retire or repair tests that assume the flat list row the
  folder tree replaced
status: To Do
assignee: []
created_date: '2026-09-09 09:13'
updated_date: '2026-09-09 09:13'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from the Notes critique fix wave (plan
Docs/superpowers/plans/2026-09-09-library-notes-critique-wave.md); raised in
the final whole-branch review of the wave, most directly by task-32128's
own lessons-file entry ("A widget id that becomes conditional must be
reconciled across all of Tests/"). Because Database Notes now composes a
folder tree by default (Agent_Lessons is always seeded), a flat
`#library-notes-row-0` row never mounts, and every test that waits on it
fails before its real assertion runs. That includes the reader suite's 13
pre-existing reds — among them
`test_database_notes_capability_inventory_and_modes` — five shell/workspace
tests, and the `filter_sort` capability node. These were red at the wave's
base too, but the wave's own reconciliation pass (task-32128) only chased
the failures its own change surfaced, not this wider pre-existing set.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every test currently asserting against `#library-notes-row-0` (or
  an equivalent flat-list assumption) either drives the tree row
  (`.library-notes-tree-note-row`) instead, or is retired with a written
  reason
- [ ] #2 The affected files' failing-test-name sets shrink accordingly,
  recorded in the task's implementation notes
- [ ] #3 No new test introduces a `#library-notes-row-N` assumption
<!-- AC:END -->
