---
id: TASK-32176
title: >-
  Library Notes: Import once copy edges leave content-less records and a
  triplicated classification set
status: To Do
assignee: []
created_date: '2026-09-09 09:14'
updated_date: '2026-09-09 09:14'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - import
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from the Notes critique fix wave (plan
Docs/superpowers/plans/2026-09-09-library-notes-critique-wave.md); raised in
the task review of task-32125/task-32130. Three copy edges survived the
Import once fix-up: a JSON array containing one content-less record reads
the whole file as "Not a note file (app configuration)" instead of
importing the valid records and reporting which one failed; the review's
**Skip all** group action is scoped to the current review page but never
says so; and the set of non-importable classifications is spelled out in
three separate places in code
(`_NON_IMPORTABLE_CLASSIFICATIONS`, `_NON_IMPORTABLE`,
`_CLASSIFICATION_LABELS`), which is one extra place to forget when the set
changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A mostly-valid structured import file reports which specific
  record failed, instead of classifying the whole file as not a note file
- [ ] #2 The **Skip all** (and equivalent group) action's label says "on
  this page"
- [ ] #3 The non-importable classification set has one source of truth,
  used everywhere it is currently duplicated
<!-- AC:END -->
