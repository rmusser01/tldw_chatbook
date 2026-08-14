---
id: TASK-16195
title: 'Study add-topic button has no handler at HEAD'
status: To Do
assignee: []
created_date: '2026-08-14 03:05'
labels:
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`#add-topic-btn` is composed in the Study surface but no handler is wired to it at current dev HEAD — pressing it does nothing. Found while verifying TASK-15471's dead-code justification for the legacy `study_events.py` handler table (which DID contain an add-topic handler; the Study rebuild moved flashcards to `Study_Modules/flashcards_handler.py` but the add-topic wiring was left behind). Decide the intended behavior (restore an add-topic flow in the current Study modules, or remove the dead button) and implement it. Related: TASK-16196 deletes the orphaned legacy handler module. Surfaced during TASK-15471 (per-click I/O off-loop, PR #1625 merged `172ada448`) and its concurrency review; evidence in the session review record.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The add-topic button either performs its intended action end-to-end or is removed from compose
- [ ] #2 A test pins whichever behavior is chosen
- [ ] #3 The decision is recorded in the notes
<!-- AC:END -->
