---
id: TASK-32177
title: >-
  Library Notes: editor leftovers — mismatched back-cue, a dead status
  widget, and a private timestamp import
status: To Do
assignee: []
created_date: '2026-09-09 09:15'
updated_date: '2026-09-09 09:15'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - editor
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from the Notes critique fix wave (plan
Docs/superpowers/plans/2026-09-09-library-notes-critique-wave.md); raised in
the task review of task-32139 and the final whole-branch review. Three
small leftovers in the Notes editor surface: the New-note view and the
load-retry view still read `‹ Notes` at compact width where task-32139's
own guide correction says `‹ Back to list`; `#library-note-context-status`
is composed on mount but then permanently hidden, dead weight in the tree;
and `_parse_browser_timestamp` is imported directly from another package's
private module rather than through a public helper.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One back-cue wording rule applies everywhere at compact width,
  including the New-note and load-retry views
- [ ] #2 The dead `#library-note-context-status` widget is removed, and the
  two geometry-pinning tests that reference it are updated
- [ ] #3 `_parse_browser_timestamp` is exposed as a public helper and the
  cross-package import is updated to use it
<!-- AC:END -->
