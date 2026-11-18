---
id: TASK-41718
title: "RAGIndexingDB cannot be backed up: its constructor adds a table capture refuses"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
labels:
- follow-up
- backup
- rag
priority: high
references:
- tldw_chatbook/DB/RAG_Indexing_DB.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41521 (round 16). `RAGIndexingDB`'s constructor adds an `embedding_cache` table, so capture refuses today's `rag_indexing.db` with `unsupported_schema`. The TASK-41521 test had to build that database from the owner's installed schema instead.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A backup of a profile whose RAG index was created by the app captures `db.rag_indexing` without `unsupported_schema`
- [ ] #2 A test creates the database through `RAGIndexingDB` and backs it up, red before the fix
<!-- AC:END -->
