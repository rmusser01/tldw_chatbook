---
id: TASK-15765
title: Repair the v17-to-v18 ChaChaNotes migration fixture broken by v35-to-v36
status: To Do
assignee: []
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

- [ ] The v17 fixture in `test_conversations_migrate_from_v17_to_v18_adds_system_prompt_column`
      also removes `note_folders` (and any note_folders-only triggers) before
      rolling `db_schema_version` back to 17, so replaying v35->v36 finds a
      genuine pre-v36 shape
- [ ] The test asserts its v17/v18 preconditions (column and trigger absence)
      before reopening at the current version, matching task-15707's pattern
- [ ] `Tests/ChaChaNotesDB/test_chachanotes_db.py` and
      `Tests/DB/test_chachanotes_console_context_memory_migration.py` pass
      with zero regressions elsewhere in `Tests/DB/` + `Tests/ChaChaNotesDB/`
- [ ] No production migration or schema code changes — this is a test-fixture
      correction only
