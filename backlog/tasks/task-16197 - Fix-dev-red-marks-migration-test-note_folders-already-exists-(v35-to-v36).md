---
id: TASK-16197
title: 'Fix dev-red marks migration test: note_folders already exists (v35 to v36)'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-14 03:05'
labels:
  - tests
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The conversation-marks migration test fails on pristine dev (`c3ed2854a` and after) with `table note_folders already exists` during the v35→v36 step — reproduced byte-for-byte on a clean base twice during TASK-15471 (implementer and reviewer independently). Same bug class as TASK-15765 (v17→v18, same error, filed 2026-08-13 after TASK-15730 fixed the earlier v33→v34 instance): a migration creates a table without guarding against its prior existence, or a fixture snapshot baked the table in early. Diagnose which side drifted (the migration or the fixture chain), fix that class-wide if the pattern generalizes — three instances now suggest the fixture-generation approach itself bakes this trap. Not attributable to TASK-15471 (pre-existing at its base); absent from known-red batch task-15766. Surfaced during TASK-15471 (per-click I/O off-loop, PR #1625 merged `172ada448`) and its concurrency review; evidence in the session review record.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The marks migration test passes on a pristine checkout
- [x] #2 Root cause named (migration vs fixture) with the introducing commit
- [x] #3 If the three-instance pattern shares one cause, the fix or a follow-up covers the class, not just this test
<!-- AC:END -->

## Implementation Plan

1. Reproduce at HEAD; verify whether the per-test repair already landed
   (task-16207 `5300077fd` added note_folders drops to the marks fixture).
2. Name the root cause and introducing commits: the fixture-generation
   approach (fresh current-version DB + hand-rolled drop list + version-stamp
   rollback) bakes future artifacts; `88f5f535a` (v33→v34 column) broke the
   lists first, `9174975b0` (v35→v36 note_folders, task-15705) broke them
   again — that commit even fixed the ONE fixture its author knew about
   (dictionary v34) and missed the other two.
3. Kill the class: shared per-version rollback registry + helper consumed by
   all three fixtures, a registry-completeness ratchet against
   `_CURRENT_SCHEMA_VERSION`, and a rollback-replay sweep across historical
   versions with schema-parity assertion against a fresh DB.
4. Mutation-verify the guard actually catches the original failure mode; run
   the marks suite, Tests/DB, Tests/ChaChaNotesDB, dictionary suite; ruff.

## Implementation Notes

The failing test is
`Tests/Chat/test_conversation_local_marks_service.py::test_local_marks_migrate_from_v16_to_v17_with_expected_schema`.
Another session had already applied the per-test repair (task-16207, commit
`5300077fd`) before this task was picked up, so it was green at base; this
task delivered the root-cause naming (AC2) and the class-wide fix (AC3).

**Root cause: the fixture chain, not the migration.** The fixture bootstraps
a FRESH DB (current version), hand-drops newer artifacts, stamps
`db_schema_version` back to 16, and reopens — so every artifact the drop
list misses is baked into the "historical" DB and collides on replay.
Introducing commits per instance:

- Instance 1 (task-15730 cluster): `88f5f535a` (V33->V34, unguarded
  `ALTER TABLE ... ADD COLUMN compaction_representation`).
- Instances 2+3 (task-15765 + this task): `9174975b0` (V35->V36
  `note_folders`, task-15705 — bare `CREATE TABLE`, unlike the IF-NOT-EXISTS
  V34->V35 and V37->V38 neighbours). Decisive class evidence: that commit
  itself patched the ONE rollback fixture its author knew about
  (`test_dictionary_attachment_index.py`) and missed the other two.

**Class fix (test layer; production migrations untouched):**

- `Tests/ChaChaNotesDB/schema_rollback.py` — a single per-version removal
  registry + `rollback_chachanotes_schema()`; all three top-down rollback
  fixtures (marks v16, chachanotes v17, dictionary v34) now consume it.
- `Tests/ChaChaNotesDB/test_schema_rollback.py` — a completeness ratchet
  (bumping `_CURRENT_SCHEMA_VERSION` without a registry entry fails by name
  with instructions) and a rollback-replay sweep over every target v16..v37
  asserting schema-object parity with a fresh bootstrap.
- The sweep caught a real latent bug during development: a defensive trigger
  drop in the V28 entry left targets V20..V27 silently missing ALL
  conversations sync triggers after replay — the parity assertion is what
  makes wrong registry entries unable to hide.
- Sweep of other fixture-built versions: only the three rollback-style
  fixtures exist (`grep "UPDATE db_schema_version"`); the bottom-up
  `Tests/DB/test_chachanotes_*_migration.py` fixtures build legacy schemas
  from scratch and are not exposed to the bake.
- Mutation evidence: emptying the v36 entry reproduces the exact filed error
  in 23 tests; deleting it trips the ratchet and the helper's actionable
  assertion.
- Verification: marks suite 19 passed; rollback guards 23 passed;
  full `Tests/DB/` + `Tests/ChaChaNotesDB/` + dictionary suite: 1215 passed,
  1 skipped, 1 pre-existing dev red (`test_current_schema_version_is_37`,
  stale contract vs v38, present on origin/dev `48ad9e7de`, unrelated).
- Lessons: incident chain recorded in
  `backlog/docs/lessons-testing-evidence.md`.

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
