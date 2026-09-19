---
id: TASK-32575
title: >-
  Library Notes: the editor heading strip clips a duplicate note's tie-break at
  100x30
status: To Do
assignee: []
created_date: '2026-09-14 22:45'
labels:
  - library
  - notes
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reproduced twice: named as a rider by wave-4 group 6 and then measured independently by the task-32558 guide sweep. task-32548 put the list row's tie-break into the editor heading so a reader who opened one of two identically titled notes could still tell which one is open. At 235x52 that works — the heading reads 'Reading list · #d3d7' (wave4-caps/docs-sweep/docs-05-editor-tiebreak-235x52). At 100x30 the strip carries the back cue on the left and the source name on the right, and the title ellipsizes between them before either: the heading reads '‹ Back to list    Reading … Library notes' and the tie-break is gone, in Edit and Preview alike (docs-08-preview-100x30, docs-09-edit-heading-clip-100x30). So on a compact terminal 'which duplicate is open' is unanswerable without going back to the list. Pre-existing. The guide now states the limit rather than the old unqualified claim; this task is the fix. A candidate the wave noted: put the answer in Info → Properties, which has room, rather than widening the strip.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On a 100x30 terminal, a reader can tell which of two identically titled notes is open without leaving the editor
- [ ] #2 Whatever carries the answer is present in Edit, Preview and Info alike, as the list row's own tie-break is
- [ ] #3 Measured live at 100x30 and 60x24 with a capture each, not only pinned
- [ ] #4 notes.md's Layout tour paragraph is updated from its current 'on a compact terminal it does not survive the heading strip' to what ships
<!-- AC:END -->
