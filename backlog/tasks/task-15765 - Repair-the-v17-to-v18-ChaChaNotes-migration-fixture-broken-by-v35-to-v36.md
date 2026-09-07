---
id: TASK-15765
title: Repair the v17-to-v18 ChaChaNotes migration fixture broken by v35-to-v36
status: Done
assignee:
  - '@claude'
created_date: '2026-08-13 12:31'
labels:
  - database
  - migrations
  - tests
priority: medium
---

## Description

**Correction to an earlier framing.** The input-latency burn-down's evidence
trail (task-15469's Implementation Notes, then task-15707/task-15730) named a
"~32 tests red" ChaChaNotes V33->V34 migration-fixture cluster tied to a
`compaction_representation` duplicate-column bug. Re-verified live on this
worktree (dev `6b57458b8`, `Tests/DB/` + `Tests/ChaChaNotesDB/`): that cluster
is gone — 1084 passed, 1 skipped, 1 failed, not 34 failed. Task-15730's v35
fixture repair evidently closed it. That original claim is stale; do not
re-file it.

**What is still genuinely red**, same fixture-shaped bug class, one instance:
`Tests/ChaChaNotesDB/test_chachanotes_db.py::TestDBInitialization::
test_conversations_migrate_from_v17_to_v18_adds_system_prompt_column` fails
with `tldw_chatbook.DB.ChaChaNotes_DB.SchemaError: Migration from V35 to V36
failed for 'rag_char_chat_schema': table note_folders already exists`.

Root cause (read from the fixture and `_migrate_from_v35_to_v36`): the test
opens a **fresh** `CharactersRAGDB` (which bootstraps all the way to the
current version, v36, via incremental migrations — so `note_folders` already
exists from that construction), then "rolls back" to a v17-shaped DB by
dropping the sync triggers, the column under test, and the newer RAG
provenance tables, and resetting `db_schema_version` to 17 before reopening.
The rollback never drops `note_folders` (added by v35->v36's
`chachanotes_v35_to_v36_note_folders.sql`, `CREATE TABLE note_folders`,
likely introduced by task-15705), so replaying v35->v36 against the reopened
DB collides with the table the fresh bootstrap already created. Same shape as
task-15707's world-book fix and task-15730's v35 fixture repairs: the
fixture's "recorded version" pointer moved back, but a later migration's
target state did not.

## Acceptance Criteria

- [x] The v17 fixture in `test_conversations_migrate_from_v17_to_v18_adds_system_prompt_column`
      also removes `note_folders` (and any note_folders-only triggers) before
      rolling `db_schema_version` back to 17, so replaying v35->v36 finds a
      genuine pre-v36 shape
- [x] The test asserts its v17/v18 preconditions (column and trigger absence)
      before reopening at the current version, matching task-15707's pattern
- [x] `Tests/ChaChaNotesDB/test_chachanotes_db.py` and
      `Tests/DB/test_chachanotes_console_context_memory_migration.py` pass
      with zero regressions elsewhere in `Tests/DB/` + `Tests/ChaChaNotesDB/`
- [x] No production migration or schema code changes — this is a test-fixture
      correction only

## Implementation Plan

1. Reproduce the red at HEAD; discover whether interim repairs landed (task-16201
   `c983320fb` already added the note_folders drops to this fixture — verify).
2. Root-cause the class with task-16197: three hand-maintained rollback drop
   lists (this test, the local-marks v16 test, the dictionary v34 test) each
   break whenever a migration adds a non-idempotent artifact.
3. Replace the per-test drop lists with a shared per-version rollback registry
   (`Tests/ChaChaNotesDB/schema_rollback.py`) used by all three fixtures.
4. Add the v17/v18 precondition assertions (column/trigger/table absence)
   before reopening, matching task-15707's pattern.
5. Add a completeness ratchet (registry must cover every version up to
   `_CURRENT_SCHEMA_VERSION`) and a rollback-replay sweep over historical
   versions so the next migration fails ONE named guard, not three fixtures.
6. Mutation-verify (remove the v36 entry → the original error returns), then
   run the named suites + Tests/DB + Tests/ChaChaNotesDB; ruff on touched files.

## Implementation Notes

By the time this task was picked up, another session had already applied the
per-test repair (task-16201, commit `c983320fb`: the fixture drops
`note_folder_memberships` + `note_folders`), so the named test was green at
base. This task therefore closed the REMAINING ACs and, with task-16197,
killed the class instead of re-patching the instance:

- Replaced the fixture's hand-rolled drop list with the new shared
  per-version rollback registry
  (`Tests/ChaChaNotesDB/schema_rollback.py::rollback_chachanotes_schema`),
  which removes everything post-v17 migrations added (incl. the note-folder
  tables) before stamping the version back to 17.
- Added the AC2 precondition assertions before reopening: `system_prompt`
  column absent, all `conversations_sync_*` triggers absent, `note_folders` /
  `note_folder_memberships` absent, recorded version == 17 (task-15707's
  pattern).
- Class guards (shared with task-16197): a registry-completeness ratchet and
  a rollback-replay sweep with schema-object parity against a fresh bootstrap
  (`Tests/ChaChaNotesDB/test_schema_rollback.py`).
- Verification: the historical red reproduced verbatim with the repair
  reverted ("table note_folders already exists" at V35->V36); mutation runs
  (v36 entry emptied / removed) fail 23 tests with the original error or the
  ratchet's actionable message. Full `Tests/DB/` + `Tests/ChaChaNotesDB/` +
  marks + dictionary suites: 1215 passed, 1 skipped, 1 failed — the single
  failure is `test_chachanotes_default_assistant_enrichment_migration.py::
  test_current_schema_version_is_37`, a PRE-EXISTING dev red (stale contract
  vs `_CURRENT_SCHEMA_VERSION == 38`; present on origin/dev `48ad9e7de`,
  untouched by this diff) — zero regressions from this change.
- No production migration or schema code changed (diff is Tests/ + backlog/
  only).

### Review follow-up (same session, pre-merge)

Independent review verdict: MERGE, two pre-merge fixes applied:

- **F1 (oracle depth)**: the sweep's parity oracle compared sqlite_master
  (type, name) only — blind to column loss, though half the registry is
  `DROP COLUMN`. `_schema_objects` now also emits a
  ("column", "<table>.<column>") entry per table column (SETS, not
  positions — F4: replay legitimately re-appends dropped columns at the
  table end). Born-red with the reviewer's exact mutation (a seeded
  `DROP COLUMN active_leaf_message_id` in entry 28): previously 22/22
  green; now exactly v24..v27 red naming the lost column, v16..v23
  repaired by the V23->V24 replay, v28..v37 unaffected. Restored
  Edit-based; unmutated registry green (23/23) — replayed column sets are
  identical to a fresh bootstrap.
- **F2 (comment truth)**: the registry docstring and both fixture comments
  no longer claim a "historical"/"genuine vN" schema. They now state what
  the fixture is — a current-version DB with the specific colliding
  artifacts removed, sufficient for replaying the migrations under test,
  NOT a faithful vN snapshot (at a v17 stamp: 7 post-v17 tables, 9
  indexes, 5 columns survive; real-vN sync triggers deliberately absent
  until replay). Precondition asserts relabelled as the fixture's own
  bake-guards. Docstring also records the F4 column-order caveat and
  points to the knowledge-free alternative (bootstrap under a patched
  `_CURRENT_SCHEMA_VERSION`, as test_chachanotes_note_folders_migration.py
  does) as the follow-up direction.
- Re-run: guards+sweep 23 passed; both formerly-red tests + dictionary
  suite green (53 passed total); ruff check/format clean.
