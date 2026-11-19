---
id: TASK-22280
title: >-
  add_message writes a v48 column, breaking every pre-v48 historical-bootstrap
  fixture (12 dev-red migration tests)
status: Done
assignee:
- '@rmusser01'
created_date: '2026-08-24'
labels:
  - testing
  - database
priority: high
dependencies: []
---

## Description

Found during task-22200 (backfill pacing), baselined against pristine dev
`983aa5878` in a clean worktree: **12 migration tests are red on dev itself**,
all with `sqlite3.OperationalError: table messages has no column named
assistant_generation_state`:

- `Tests/DB/test_chachanotes_v47_messages_fts_backfill.py` — 8 of 14 red
  (including the SIGKILL resumability witness and both snapshot-upgrade
  interleave tests for the hot-writer IMMEDIATE contract)
- `Tests/DB/test_chachanotes_sync_log_retention_migration.py` — 4 of 7 red

Mechanism: `CharactersRAGDB.add_message` now unconditionally INSERTs
`assistant_generation_state` (the sync/continuation work around `a2e056a8d` /
`dc10bda0a` / `b89052dc4`), but that column is only created by the v47->v48
migration (`chachanotes_v47_to_v48_console_library_policy.sql:22`). Every
fixture that builds a genuinely historical DB via
`Tests/ChaChaNotesDB/historical_bootstrap.chachanotes_db_at_version(path, 44/45)`
and then seeds through `add_message` (`_seed_v45` and friends) fails at the
seed, so the migration guarantees those files existed to pin (deferred FTS
rebuild, kill-safety, rewind-on-failure, the v47 trigger guards) currently
have no working witnesses.

Note the coverage shape when fixing: task-22200's pacing tests deliberately
sidestep this by seeding at CURRENT schema + the migration's own
`'delete-all'` reset (see `Tests/DB/test_chachanotes_fts_backfill_pacing.py`'s
module docstring), which reproduces the backfill WINDOW but cannot replace the
v45-replay coverage these red tests provided. The likely fix directions:
either `add_message` probes/branches on column presence (ugly, production code
serving tests), or the historical seeding helpers write pre-v48 rows with
version-appropriate SQL instead of calling today's `add_message` — the
historical-bootstrap module's own philosophy ("knowledge about the single
migration the test pins, owned by that test") points at the latter.

## Acceptance Criteria

- [x] The 12 enumerated tests pass on dev without weakening what they pin (the deferred-rebuild, kill-safety, rewind, and trigger-guard assertions stay intact)
  - Satisfied by TASK-21441 (PR #2082, commit `9548de4165`, 2026-08-25 — one day after
    this task was filed), which measured these files going red-to-green on base
    `a71e62e4b` (13 failed in Tests/DB + 1 in Tests/ChaChaNotesDB + 2 packaging
    parametrizations → 0 failed / 1710 passed / 1 skipped); none of the pinned
    assertions were weakened. The files have SINCE gone red again (8 deterministic
    failures at `e92b01515f`, 2026-10-02) from LATER, separately-filed drift that is
    NOT this task's mechanism — `console_trace_graph_epoch` (a v56 TABLE written by
    `soft_delete_message`/`update_message` seeding, owned by **TASK-33371**) and the
    HOT_MESSAGE_WRITERS interleave/static-guard blindness after `add_message` became
    a thin wrapper (owned by **TASK-33621.36**). This task closes as
    already-fixed-by-predecessor for its named cause; re-broken-state ownership lives
    with those two tasks.
- [x] Seeding a `chachanotes_db_at_version(..., 44/45)` fixture with messages works again, via a mechanism that does not require production `add_message` to know about test schemas
  - `_messages_insert_statement` (ChaChaNotes_DB.py:13270) derives the INSERT column
    list from `PRAGMA table_info(messages)` per instance — schema introspection, not
    test-schema knowledge — dropping an absent column only when its value is None and
    raising `SchemaError` otherwise. Verified live at `e92b01515f`: `add_message`
    seeds v44 and v45 fixtures directly (witness in Implementation Notes).
- [x] A guard or lesson entry records how a production column addition silently invalidated historical-bootstrap seeding, so the next `messages` column addition does not repeat this
  - Guard: `Tests/DB/test_chachanotes_bare_open_self_migration.py` pins the
    per-schema writer, including `pytest.raises(SchemaError, match="assistant_generation_state")`
    for an absent column carrying data (line 463) and direct
    `_messages_insert_statement` assertions (line 490); green on this base. The
    `_messages_insert_statement` docstring records the incident and the
    recurrence-by-construction argument. Lesson:
    `backlog/docs/lessons-testing-evidence.md` entry "A new schema artifact
    maintained ON WRITE breaks every historical-bootstrap fixture at once"
    (TASK-32186, 2026-09-11) prescribes checking `historical_bootstrap` in the same
    commit that adds an on-write-maintained artifact.

## Implementation Plan

1. Verify the premise at the current dev base (`e92b01515f`): run the two named
   files, capture per-test tracebacks, and attribute each red to a mechanism.
2. Confirm whether the named cause (add_message writing v48's
   `assistant_generation_state` into pre-v48 fixtures) still reproduces; if a
   predecessor already fixed it, gather closure evidence (commit, guard test,
   lesson entry, direct seeding witness).
3. Sweep the board for tasks already owning any residual reds before touching code.
4. Close the task file with the evidence and cross-links; one commit; no
   production-code changes unless the premise turned out live.

## Implementation Notes

**Verdict: already fixed by predecessor — closed with evidence, no code changed.**

The named defect (`add_message` unconditionally INSERTing v48's
`messages.assistant_generation_state`, breaking every pre-v48
historical-bootstrap seed) does not exist on this base. It was fixed one day
after filing by TASK-21441 / PR #2082 (commit `9548de4165`, 2026-08-25, an
ancestor of `e92b01515f`): `_add_message_with_semantic_sidecars` now builds its
INSERT through `_messages_insert_statement`, which reads
`PRAGMA table_info(messages)` once per instance, writes only columns the table
actually has, drops an absent column only when its value is `None` (provably
lossless — the NULL it would have received), and raises `SchemaError` when an
absent column carries data. That is the per-schema, no-per-bump shape this
task's Description anticipated ("the repair is per-schema rather than
per-column").

**Evidence (all commands run in this worktree at `e92b01515f`, venv
`python3.12`, `uv pip install -e ".[dev]"`):**

- Premise check, first (cold, loaded machine) run:
  `python -m pytest Tests/DB/test_chachanotes_v47_messages_fts_backfill.py Tests/DB/test_chachanotes_sync_log_retention_migration.py -q`
  → 9 failed / 12 passed.
- Deterministic set on rerun (serial and xdist): **8 failed** — 5 in the v47
  FTS-backfill file, 3 in the sync-log-retention file
  (`test_backfill_survives_sigkill_mid_run` passed on rerun; its single
  first-run failure was kill-timing flake under shared-CPU load, same traceback
  class as the seed reds below).
- `add_message` is NOT the failure point in any of them: the seed logs show
  `Added message ID ... to conversation ...` immediately before each failure,
  and the tracebacks land in the seeds' NEXT calls:
  `_seed_v45`/retention seed → `soft_delete_message` (ChaChaNotes_DB.py:16016)
  / `_update_message_uncoordinated` (:15351) → `_advance_semantic_graph_epoch`
  (:13668) → `sqlite3.OperationalError: no such table:
  console_trace_graph_epoch` — a v55→v56 TABLE
  (`chachanotes_v55_to_v56_console_semantic_trace.sql:370`), absent from v44/v45
  fixtures. Two further failures are the interleave tests' `execute_query` hook
  never firing (the INSERT moved to `conn.execute` inside `transaction()` in
  the same PR #2082 refactor) and the static
  `test_hot_message_writers_reserve_the_write_lock_up_front` failing because
  `add_message` is now a thin wrapper delegating to
  `_add_message_with_semantic_sidecars`, so `inspect.getsource(add_message)`
  no longer contains `self.transaction(immediate=True)`.
- Direct AC#2 witness (with `Tests/conftest.py`'s sandbox imported to avoid the
  known `RecoveryRequired` config-admission trip):
  `chachanotes_db_at_version(p, 44/45)` + `add_conversation` + `add_message`
  → both printed `add_message -> '<uuid>', messages=1, schema_version=44/45`.
- Guard/AC#3 witness:
  `python -m pytest Tests/DB/test_chachanotes_bare_open_self_migration.py Tests/ChaChaNotesDB/test_historical_bootstrap.py -q`
  → 85 passed.
- Board sweep (before any fix design): the residual reds are already owned —
  **TASK-33371** (To Do, 2026-09-28) names "Migration fixtures are missing
  tables (console_trace_graph_epoch, note_links)" and its AC triages each as
  test-drift vs product defect; **TASK-33621.36** (To Do, 2026-09-30) names
  "The HOT_MESSAGE_WRITERS write-lock guard is blind, and 5 v47 FTS backfill
  tests are red". Fixing either here would duplicate an open task's scope; the
  epoch gap additionally needs a design choice (make the shipped v55→v56
  `CREATE TABLE` idempotent vs `_add_forward_write_dependencies` pre-creation
  per the TASK-32186 lesson vs version-appropriate seed SQL) that TASK-33371's
  triage AC explicitly reserves.

**Approach taken:** verification-and-attribution closure. No production or test
code changed. Cross-reference lines added to TASK-33371 and TASK-33621.36
pointing at this closure's measurements, and a hygiene lesson recorded (see
below).

**Trade-off:** AC#1's literal today-state (the enumerated files green at
closure time) is false — deliberately not "fixed" here because every current
red belongs to a different, later mechanism with existing board ownership;
re-fixing under this task would create colliding ownership of the same reds.

**Modified or added files:** `backlog/tasks/task-22280 - ...md` (this file),
`backlog/docs/lessons-backlog-hygiene.md` (one lesson),
`backlog/tasks/task-33371 - ...md` and
`backlog/tasks/task-33621.36 - ...md` (one-line cross-references only).

ADR required: no
Reason: documentation-only closure of a stale-premise task; no schema, sync,
boundary, or contract decision made or changed. The one design-relevant fact
(the epoch gap's candidate fix shapes) is recorded here for TASK-33371's owner
instead.

**Lesson recorded:** `backlog/docs/lessons-backlog-hygiene.md` — "A stale
premise can be fixed by a predecessor while its enumerated tests re-break for
newer reasons" (state the incident: this task, 2026-10-02).
