---
id: TASK-41513
title: "RAG's own Review recovery never approves rag.definitions, rag.projections or db.rag_indexing"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- backup
- rag
priority: high
references:
- tldw_chatbook/Backup_Recovery/recovery_service.py
- Tests/Backup_Recovery/test_rag_model_recovery_controls.py
dependencies:
- TASK-41301
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the bkreview fix round (round 15) once the test's line-121 failure was fixed (TASK-41301). The `[rebuild]` node of `test_rag_model_recovery_controls.py` shows that RAG's own Review recovery never approves `rag.definitions`, `rag.projections` or `db.rag_indexing`; it was red at base, hidden behind the earlier failure. A user who follows the RAG recovery path cannot finish it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Following RAG's own Review recovery approves the owners it needs and the RAG index becomes usable
- [ ] #2 The `[rebuild]` node passes, red before the fix
- [ ] #3 The review copy names what was approved
<!-- AC:END -->
