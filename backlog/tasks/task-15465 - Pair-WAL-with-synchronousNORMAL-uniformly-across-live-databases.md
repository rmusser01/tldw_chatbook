---
id: TASK-15465
title: Pair WAL with synchronous=NORMAL uniformly across live databases
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
updated_date: '2026-08-11 20:27'
labels:
  - perf
  - db
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Probe-verified in the audit (effective PRAGMAs read from created DB files): eight post-2025 databases run at journal_mode=DELETE + synchronous=FULL — Subscriptions (`DB/Subscriptions_DB.py:184-196`), Workspace (`:37-40`), Library_Collections (`:21-24`), RAG_Indexing, search_history, client_notifications, scheduled_tasks (Research/Writing are dead code, see task-15481) — and five WAL databases never set synchronous=NORMAL, so every commit still fsyncs the WAL: ChaChaNotes (`:2767-2769`), Client_Media (`:737-738`), Prompts (`:438-439`), Evals (`:191-192`), AgentRuns (`:73-75`). DELETE mode additionally makes writers exclusive-lock readers; multiple connections contend on the Subscriptions file with sqlite's default 5.0 s busy timeout — a direct candidate for the reported multi-second stalls on slow disks. `DB/Library_Ingest_Jobs_DB.py:57-61` already does it right (WAL + NORMAL with in-line rationale) and is the template.

Stability notes: WAL+NORMAL is the SQLite-documented safe pairing (app-crash-safe; an OS/power crash may lose the last commits — acceptable for these local caches/stores, document per DB); WAL creates -wal/-shm sidecars, so backup/maintenance surfaces that reference DB file paths must be checked (task-899 history); skip the dead DB modules rather than churn them. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every live application database runs WAL + synchronous=NORMAL, verified by a probe reading effective PRAGMAs from created files
- [x] #2 A write-stall microbenchmark on one write-hot path (e.g. watchlists item upserts or workspace pin/save) recorded before/after
- [x] #3 Backup/restore and DB-maintenance surfaces handle -wal/-shm sidecars correctly (test or documented check)
- [x] #4 Each DB carries a one-line rationale comment like the Library_Ingest_Jobs template
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read every target DB's connection-open site(s) (held thread-local, per-op fresh-connection, and memory-only patterns differ per file) to find where journal_mode/synchronous must be paired.
2. DELETE+FULL -> WAL+NORMAL: Subscriptions_DB, Workspace_DB, Library_Collections_DB, RAG_Indexing_DB, client_notifications_db, scheduled_tasks_db (the last has no _get_connection override today -- add one). Guard journal_mode=WAL with is_memory_db where the file already does so elsewhere; synchronous=NORMAL is set unconditionally (harmless on :memory:).
3. WAL-but-FULL -> WAL+NORMAL: add a synchronous=NORMAL pragma next to the existing journal_mode=WAL call in ChaChaNotes_DB, Client_Media_DB_v2, Prompts_DB, Evals_DB, AgentRuns_DB.
4. Add a one-line rationale comment per edited connection site, matching the Library_Ingest_Jobs_DB.py:57-61 template shape.
5. Write Tests/DB/test_pragma_settings.py (new, committed): construct each live DB class against a tmp_path file, read back PRAGMA journal_mode / PRAGMA synchronous, assert wal/NORMAL(1); also cover the :memory: case for classes that support it, asserting no exception and journal_mode=='memory'.
6. Run targeted existing suites for every touched DB plus the new test file.
7. Write an isolated scratch-HOME write-stall microbenchmark (before/after) against one write-hot path (watchlists item upsert via SubscriptionsDB or WorkspaceDB pin/save), not committed to the repo; record numbers in task notes and the report.
8. Audit backup/restore/VACUUM/maintenance surfaces (Tools_Settings_Window.py, private_sqlite.py) for -wal/-shm sidecar handling; document findings (fix only if a real gap is found).
9. Self-review diff, tick ACs, add Implementation Notes, commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
FIX ROUND 2 (scoped re-review): Sync_Interop/notes_mirror.py's NotesMirror
had production references and its own test suite but opened its held
connection with no journal pragmas at all -- same profile as Tamagotchi
(round 1). Paired WAL+NORMAL, added to the pragma test (41 cases now, was
39). Swept every connect_private_sqlite/sqlite3.connect production call
site by grep (not sampling) to make the pragma test's "every live
application database" docstring claim exhaustive; found no further
gaps. Documented the two legitimate exclusion categories (read-only-by-
construction connections; backup/restore/migration/candidate connections
that only ever touch a disposable temp-file copy or reopen the live store
read-only-in-practice during a rare migration) directly in the docstring
with owner ids, plus re-confirmed via grep the five dead-code DB modules
this task's own description already names as an explicit skip have no
import site anywhere. Tests/DB/test_pragma_settings.py: 41 passed.
Tests/Sync_Interop/test_notes_mirror.py: 4 passed. Tests/Sync_Interop/
(full): 215 passed. Commit e36d9adcd.
<!-- SECTION:NOTES:END -->
