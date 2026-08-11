---
id: TASK-15465
title: Pair WAL with synchronous=NORMAL uniformly across live databases
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
updated_date: '2026-08-11 20:15'
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
FIX ROUND 1 (review): AC#1's "every live application database" was not yet
true -- 8 live SQLite stores outside the original audit's DB/ enumeration
were unconverted. Converted all 8 to WAL+NORMAL at every connection-open
site, same is_memory-guarded convention, rationale comment each:
Kanban_Interop/local_kanban_db.py, TTS/profile_schema.py (3 sites in
open_profile_store), Writing_Interop/local_writing_service.py,
Research_Interop/local_research_service.py,
Notifications/event_state_repository.py, Sync_Interop/sync_state_repository.py,
Notes/file_notes_replica.py, Widgets/Tamagotchi/tamagotchi_storage.py
(no import site found outside its own module -- converted anyway per
review instruction, dormancy noted in its docstring).

Minors fixed same round: Subscriptions_DB.py's ensure_site_configs_schema
now pairs WAL+NORMAL (previously opened with no pragmas at all); added
Library_Ingest_Jobs_DB (the template) to the pragma regression test;
RAG_Indexing_DB.py's comment now documents the WAL lingering-reader/
unbounded -wal-growth failure mode and points at task-15466; Utils/
sensitive_paths.py's _DB_PATH_ACCESSOR_NAMES gained the missing
get_evals_db_path/get_rag_indexing_db_path entries (both honor [database]
overrides like every other listed accessor; no get_agent_runs_db_path
accessor exists since that DB's path always derives from ChaChaNotes').

Tests/DB/test_pragma_settings.py now covers 19 classes/functions (39
parametrized cases: 20 file-backed + 19 memory -- tts.profile_store has no
:memory: target per its private-file-only owner policy). All 39 pass.
Targeted suites for every newly touched store + the Subscriptions suite:
1593 passed, 9 skipped, 0 failed. 3 pre-existing, confirmed-unrelated
failures in Tests/TTS/test_tts_profile_capabilities.py (a Protocol
isinstance rejection unrelated to SQLite) left untouched per review
instruction. Commit a2c1298a2.
<!-- SECTION:NOTES:END -->
