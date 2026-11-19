---
id: TASK-41954
title: "Other ChaChaNotes tests red on dev: sargable timestamps, v82/v83 portrait migrations, message cascade deletes"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- tests
- db
priority: low
references:
- Tests/ChaChaNotesDB
dependencies:
- TASK-41655
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** Seen during TASK-41655: `test_sargable_timestamps` pins version 80; the v82 and v83 portrait migration tests fail; two message cascade-delete tests are refused with "semantic mutation authorization required".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each node has a recorded cause and is repaired (drift) or fixed red-first (defect)
<!-- AC:END -->
