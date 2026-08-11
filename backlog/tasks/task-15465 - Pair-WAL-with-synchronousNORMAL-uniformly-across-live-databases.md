---
id: TASK-15465
title: Pair WAL with synchronous=NORMAL uniformly across live databases
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
updated_date: '2026-08-11 19:54'
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
Paired journal_mode=WAL with synchronous=NORMAL on every live application
DB's connection-open site(s), matching the Library_Ingest_Jobs_DB.py:57-61
template.

DELETE+FULL -> WAL+NORMAL: Subscriptions_DB, Workspace_DB,
Library_Collections_DB, RAG_Indexing_DB, client_notifications_db (both the
file and memory branches), scheduled_tasks_db (added a _get_connection
override -- it had none before, inheriting BaseDB's plain row-factory-only
connection).

WAL-but-FULL -> WAL+NORMAL: ChaChaNotes_DB, Client_Media_DB_v2, Prompts_DB,
Evals_DB, AgentRuns_DB -- one added PRAGMA line next to the existing
journal_mode=WAL each already had.

Every edit carries a one-line rationale comment (crash-safety trade-off +
why synchronous must be re-applied per connection for the fresh-per-op
classes). is_memory_db guards WAL only, matching each file's existing
convention; synchronous=NORMAL is applied unconditionally (harmless on
:memory:).

AC#1: new Tests/DB/test_pragma_settings.py constructs all 11 classes against
tmp_path files AND :memory:, asserting wal+NORMAL(1) on file and 'memory'
(no raise) on :memory:. 22 cases, all pass.

AC#2: scratch microbenchmark (isolated HOME/XDG/TLDW_CONFIG_PATH, not
committed) drove 300 single-item commits through the real
persist_subscription_item upsert path (SubscriptionsDB, the watchlist/feed
item write-hot path) under DELETE+FULL vs WAL+NORMAL. Two runs:
run 1: DELETE+FULL mean 0.389ms/commit (total 116.7ms) vs WAL+NORMAL mean
0.115ms/commit (total 34.6ms) -- 3.4x.
run 2: DELETE+FULL mean 0.397ms/commit (119.2ms) vs WAL+NORMAL mean
0.121ms/commit (36.2ms) -- 3.3x.
(Fast M-series SSD; DELETE+FULL's per-commit fsync cost is disk-bound and
would be larger on slower/networked storage -- consistent with the audit's
"multiply 3-5x for constrained hardware" note.)

AC#3: audited backup/restore/VACUUM in UI/Tools_Settings_Window.py. All of
it (copy_private_sqlite/restore_private_sqlite/connect_private_sqlite) funnels
through DB/private_sqlite.py's centralized seam, which already treats
-wal/-shm/-journal as first-class sidecars (_SIDECAR_SUFFIXES, applied in
both _prepare_source_artifacts for backup sources and
_connect_registered_sqlite for every open) and uses SQLite's own backup API
(never a raw file copy) -- so WAL content gets checkpointed correctly rather
than needing sidecar files copied at all. Confirmed via grep no raw
shutil.copy* bypass for any of the 11 DBs. Workspace_DB, Library_Collections_DB,
AgentRuns_DB, ScheduledTasksDB, and ClientNotificationsDB have no Settings
backup/restore/VACUUM surface at all today -- a pre-existing coverage gap,
not a sidecar-correctness bug, and out of scope here. No code changes needed
for AC#3.

AC#4: every edited connection-open site carries its own one-line rationale
comment (see diff).

Testing: Tests/DB/test_pragma_settings.py (22 passed) + targeted suites for
every touched DB (Subscriptions, Workspace, Library_Collections, RAG_Indexing,
client_notifications, scheduled_tasks, Media, Prompts, Evals, AgentRuns,
private_sqlite inventory/interop) = 1738 passed, 8 skipped, 0 failed.

Pre-existing, unrelated finding: Tests/ChaChaNotesDB + 12 Tests/DB/
test_chachanotes_*_migration.py files show 33 failures / 284 passed, all
tracing to the same root cause -- "Migration from V33 to V34 failed for
'rag_char_chat_schema': duplicate column name: compaction_representation"
in ChaChaNotes_DB.py's V33->V34 migration (chachanotes_v33_to_v34_visual_
compaction_policy.sql, TASK-14914). Confirmed via a manual revert-and-retest
of the synchronous=NORMAL line that this is 100% present on dev at c0c4753f8
independent of this task's change (a plain fresh CharactersRAGDB() from v0
reaches v34 fine; the failure only triggers on the "rewind an existing DB to
an older version, then reconstruct" test shape used throughout that
directory). Not fixed here -- out of scope for task-15465, flagging for
whoever owns TASK-14914/the migration chain.
<!-- SECTION:NOTES:END -->
