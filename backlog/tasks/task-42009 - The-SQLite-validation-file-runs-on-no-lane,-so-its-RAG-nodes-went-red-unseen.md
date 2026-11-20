---
id: TASK-42009
title: "The SQLite validation file runs on no lane, so its RAG nodes went red unseen"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- tests
- backup
priority: low
references:
- Tests/Backup_Recovery/test_sqlite_validation.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41718 (round 17). `test_sqlite_validation.py`'s two `test_native_rag_indexing_schema_zero_survives_restricted_validation` nodes were red on dev until TASK-41718 fixed the RAG schema check; nothing noticed because the file runs on no lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The file runs on a PR or dev-push lane, with its time recorded
<!-- AC:END -->
