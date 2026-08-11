---
id: TASK-15466
title: Port the held-connection idiom to Library_Collections, RAG_Indexing, and client_notifications
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - db
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: three wired DBs still open a fresh sqlite connection per operation — the pre-task-3011 anti-pattern. `DB/Library_Collections_DB.py:26-43` (`closing()` per op; its service also opens write-`BEGIN` transactions for pure reads at `Library/library_collections_service.py:380/:450/:519`; hot on the Library screen, though off-loop via to_thread); `DB/RAG_Indexing_DB.py:102/:136` (fresh connection PER INDEXED ITEM in `mark_item_indexed`, and `with self._get_connection()` is a transaction context manager so connections are never closed, only GC'd — plus a per-item fsync at DELETE/FULL); `Notifications/client_notifications_db.py:56+` (same never-closed shape, insert per dispatched notification). Each fresh connect costs ~2x a raw connect because the private-sqlite seam re-verifies the DB file and three sidecars per open (`DB/private_sqlite.py:948-991`).

Fix direction: the Workspace_DB held thread-local idiom — WITH `isolation_level=None` (the task-3012 lesson: without it, sqlite3 auto-BEGINs on DML and the accumulated implicit transaction breaks explicit BEGIN and silently rolls back bare DML on close). Mandatory pre-port step per that lesson: audit every `connection()` call site for bare DML and record the audit in this task. Pure-read paths stop opening write transactions. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All three DBs hold thread-local connections with isolation_level=None; the pre-port DML call-site audit is recorded in this task
- [ ] #2 Pure-read service calls no longer open write transactions (evidence)
- [ ] #3 RAG indexing no longer opens/commits per item — batched (evidence)
- [ ] #4 All three DBs' existing test surfaces green; no connection leak (probe or unit evidence)
<!-- AC:END -->
