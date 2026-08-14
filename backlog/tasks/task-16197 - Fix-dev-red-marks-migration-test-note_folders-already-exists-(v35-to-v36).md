---
id: TASK-16197
title: 'Fix dev-red marks migration test: note_folders already exists (v35 to v36)'
status: To Do
assignee: []
created_date: '2026-08-14 03:05'
labels:
  - tests
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The conversation-marks migration test fails on pristine dev (`c3ed2854a` and after) with `table note_folders already exists` during the v35→v36 step — reproduced byte-for-byte on a clean base twice during TASK-15471 (implementer and reviewer independently). Same bug class as TASK-15765 (v17→v18, same error, filed 2026-08-13 after TASK-15730 fixed the earlier v33→v34 instance): a migration creates a table without guarding against its prior existence, or a fixture snapshot baked the table in early. Diagnose which side drifted (the migration or the fixture chain), fix that class-wide if the pattern generalizes — three instances now suggest the fixture-generation approach itself bakes this trap. Not attributable to TASK-15471 (pre-existing at its base); absent from known-red batch task-15766. Surfaced during TASK-15471 (per-click I/O off-loop, PR #1625 merged `172ada448`) and its concurrency review; evidence in the session review record.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The marks migration test passes on a pristine checkout
- [ ] #2 Root cause named (migration vs fixture) with the introducing commit
- [ ] #3 If the three-instance pattern shares one cause, the fix or a follow-up covers the class, not just this test
<!-- AC:END -->
