---
id: TASK-19553
title: >-
  Old-style ChaChaNotes migration steps are non-atomic and can permanently brick
  a database
status: Done
assignee: []
created_date: '2026-08-21 20:03'
labels:
  - db
  - migrations
  - data-loss
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 3 (data layer & schema integrity) —
its **F1**, and the lane's only finding rated SEVERE. **CONFIRMED BY LIVE
EXPERIMENT**, not by reading.

The old-style migration steps in `tldw_chatbook/DB/ChaChaNotes_DB.py` are
**non-atomic and non-idempotent**, so a partial apply leaves the database in a
state that can never be repaired from inside the app.

Mechanism: `executescript` commits the pending transaction and then autocommits
each statement individually. **22 of 38 steps use `executescript`, and 22 of 38
have no entry-version guard**, several with bare unguarded DDL — the v12→v13
step is 13 `ALTER`s and 4 `CREATE TRIGGER`s with nothing to make a re-run safe.

The lane's live experiment: it ran the chain on a genuine v11 database with one
of four columns already applied. Two `ALTER`s **stayed committed** while the
schema version stamp **stayed at 11**; "rolled back cleanly" evaluated to
`False`. Every subsequent launch re-raises on the already-applied column,
`CharactersRAGDB.__init__` fails, and conversations, notes and characters all
become unreachable — with **no in-app recovery path**. The control case (a
new-style step using `self.transaction()`) rolled back cleanly.

Reachability is not hypothetical and is not the newest users: databases in the
field sit at v4–v25, i.e. exactly the users who must replay the longest chain
to reach the current `_CURRENT_SCHEMA_VERSION` (42).

**Fix (the lane's, and it fits the owner's durable-over-clever standing
ruling):** port the remaining `executescript` steps to the
`transaction()` + per-statement pattern already used at
`ChaChaNotes_DB.py:5203`. That pattern is in-repo, proven by the control, and
does not require inventing anything.

Note the version-guard machinery itself is *not* broken — the lane verified
that the `executescript`-commits-the-transaction hazard is explicitly handled
where the code re-issues `BEGIN`. The defect is the steps that never adopted
the safe pattern.

## Acceptance Criteria

- [x] Every migration step in `ChaChaNotes_DB.py` applies atomically: an
      interrupted or failing step leaves the database at its entry version with
      no partially-applied DDL
- [x] Every step is idempotent or entry-version guarded, so re-running a step
      after an interruption is safe rather than fatal
- [x] The remaining `executescript` steps are ported to the
      `transaction()` + per-statement pattern at `ChaChaNotes_DB.py:5203`
      (durable, already proven in-repo) rather than a new mechanism
- [x] A test reproduces the lane's experiment: a genuine older-version database
      with one column of a multi-`ALTER` step pre-applied migrates to current
      **or** rolls back cleanly to its entry version — it never lands
      half-applied with a stale version stamp
- [x] The test covers the long-chain case (a v4-era database replayed to
      current), since that is the reachable population
- [x] If any already-bricked shape is recoverable, a recovery path exists that
      does not require the user to hand-edit SQLite

## Implementation Plan

1. Reproduce the lane's brick as a runnable probe (genuine v11 DB, one
   statement of V11→V12 pre-applied) and record the exact failure signature.
2. Build a byte-identity oracle BEFORE editing: snapshot every bootstrap
   version 4..42 plus the chain replay plus a fresh build, capturing verbatim
   `sqlite_master.sql` and `PRAGMA table_info` including column order.
3. Add a shared statement runner (split on `sqlite3.complete_statement`, one
   `cursor.execute` per statement) mirroring `_migrate_from_v37_to_v38`.
4. Port every `executescript` migration step to it, wrapped in
   `self.transaction()`; add the entry-version guard where missing.
5. Add the two idempotence rules that make an already-half-applied step
   re-enterable, both no-ops on a healthy chain.
6. Re-run the oracle and require ZERO divergence; mutation-test the oracle so
   the zero means something.
7. Land the born-red regression suite; baseline every test gate against
   `origin/dev` before attributing any red.

## Implementation Notes

Ported all 25 `conn.executescript(...)` migration steps — plus
`_apply_schema_v4`, the v4 base apply — to the
`with self.transaction() as cursor:` + one-`cursor.execute`-per-statement
pattern already proven by `_migrate_from_v37_to_v38`. `executescript` commits
the caller's transaction and then autocommits each statement, which is exactly
why a half-applied step used to strand a database with committed DDL under a
stale version stamp. No step was renumbered or merged and
`_CURRENT_SCHEMA_VERSION` is untouched.

**New shared machinery** (`tldw_chatbook/DB/ChaChaNotes_DB.py`):

* `_split_sql_statements` / `_strip_leading_sql_noise` (module level) — the
  splitter the new-style steps already used for their `.sql` files, now shared;
  leading comments are stripped only for MATCHING, never for execution, so
  `sqlite_master.sql` text is untouched.
* `_execute_migration_statements(cursor, script, label)` — the runner.
* `_require_migration_entry_version(conn, expected, label)` — the guard the
  new-style steps carry, now on every step.
* Two idempotence rules, both provably no-ops on a healthy chain (the column /
  trigger can only pre-exist on an already-damaged database):
  `_skip_already_applied_add_column` (SQLite has no `ADD COLUMN IF NOT EXISTS`;
  this is the same guard v19→v20 and v29→v30 already hand-rolled, generalised)
  and `_drop_superseded_trigger` (V7→V8 and V8→V9 create ~19 triggers with
  neither `IF NOT EXISTS` nor a preceding `DROP`; statements that DO say
  `IF NOT EXISTS` are deliberately left alone so SQLite's keep-the-existing-one
  semantics are not overridden).

**What porting `_apply_schema_v4` did and did NOT buy** (corrected in the fix
round — the first version of this note claimed a brick that does not exist).
The v4 base script is already re-enterable on its own terms: 42
`CREATE TRIGGER` but 42 matching `DROP TRIGGER IF EXISTS` (zero creates
without a preceding drop), every `CREATE TABLE`/`VIRTUAL TABLE`/`INDEX`
carrying `IF NOT EXISTS`, both top-level inserts `INSERT OR IGNORE`. Sweeping
all 120 interruption points of the script at the merge base, the retry reached
version 42 in **120 of 120** cases. The real, measured benefit is leftover
state: at the worst point the pre-fix path committed **111 `sqlite_master`
rows** into a file the caller was told had failed to initialize (119 of 120
points left something), versus **0 at every point** after. It also removes the
last `executescript` from the schema path, making "no step commits"
unconditional. The genuine bare-trigger hazard is in the MIGRATION steps, and
it is exactly two of them: V7→V8 (8) and V8→V9 (13), 21 unprotected
`CREATE TRIGGER`s in total — reproduced red on the pre-fix code by
`test_interrupted_trigger_step_recovers`.

**Trade-off worth knowing: one long write transaction.** Because no step
commits any more, an upgrade is a single transaction spanning the whole chain
instead of dev's ~38 per-step commits. On a very large v4-era database that
means more WAL growth before the single commit, and the write lock is held for
the whole upgrade rather than released between steps. That is the right price
for atomicity — partial progress through a migration chain has no value, since
a database stranded at an intermediate version is unusable by current code
either way — but it is a real change in resource profile, not a free win, and
the WAL only checkpoints back down after the commit. Contention is not a
practical concern here (this runs inside `CharactersRAGDB.__init__`, before
the app has any other user of the connection), and `_initialize_schema`'s
`BEGIN` is DEFERRED exactly as before, so the read-then-write upgrade window
is unchanged from dev.

**One statement deliberately left outside a transaction.**
`_FULL_SCHEMA_SQL_V4` opens with `PRAGMA foreign_keys = ON`, which SQLite
silently IGNORES inside a transaction. It is not forced: the guarantee is
carried by `_get_thread_connection`, which issues the same pragma on every
connection before the schema apply runs. Verified live (fresh DB reports
`PRAGMA foreign_keys = 1` and a dangling-FK insert raises `IntegrityError`) and
pinned by `test_foreign_keys_remain_enforced_on_a_fresh_database`.

**Deliberate behaviour change.** With no step committing, the whole chain is
one transaction, so a failure at step N rewinds to the RUN's entry version, not
to step N-1. `Tests/DB/test_chachanotes_citation_provenance_migration.py`'s
`test_citation_failure_after_dev_migrations_leaves_clean_v26` had pinned the
old partial-commit behaviour; it is rewritten (and renamed `..._v24`) to assert
whole-run atomicity plus the property that actually matters — the rewound
database still opens and migrates on the next attempt.

**Evidence.**

* Byte identity: a snapshot of all 39 bootstrap versions (v4..v42) + the
  v4→current chain replay + a fresh in-memory build, capturing verbatim
  `sqlite_master.sql` and `PRAGMA table_info` including `cid`, taken before the
  edit and diffed after — 11,177 objects, 22,815 column entries, **0
  divergences**. The oracle was mutation-tested so that zero means something:
  deleting one `CREATE INDEX` from a step → 40 divergences; swapping two
  `ADD COLUMN` statements (order only) → 64.
* Born-red: 12 of the 21 new tests fail on `origin/dev` (with local copies of
  the two new text helpers so they fail on BEHAVIOUR, not `ImportError`); all
  21 pass here.
* `Tests/ChaChaNotesDB/`: 298 passed / 3 failed — the same 3 failures
  `origin/dev` has (277/3 there; +21 new). `Tests/DB/`: 1045 passed / 9 failed
  — identical to `origin/dev`'s 1045/9. Focused consumer sweep (12 files
  incl. packaging + notes sync + chatbooks): 488 passed / 5 failed on BOTH.
  Repo-wide `pytest --collect-only -q`: 53,650 collected, exit 0.

**Fix round (post-review).** The reviewer endorsed the code change and
adjudicated the whole-run-atomicity behaviour change in its favour, but caught
a **false incident recorded as fact** in a permanent code comment: the claim
that the v4 base apply's bare `CREATE TRIGGER`s made a retry die on "trigger
already exists". I re-measured and confirmed the reviewer: it never happens
(120/120 retries recovered pre-fix). Corrected at all three sites
(`ChaChaNotes_DB.py:_apply_schema_v4`, and the test's docstring and assertion
message) to state the measured leftover-state benefit instead. Why it survived
is the instructive part: the test asserts *leftovers are zero*, which the
pre-fix code genuinely fails, so it reds before ever reaching the retry — the
claimed failure mode was never actually observed, and a green test validated
the assertion, not the story told about it.

Also in the fix round: `test_every_shipped_migration_script_splits_cleanly`
now asserts each chunk holds **exactly one** statement (359 statements across
all embedded scripts plus the base script), with its own mutation control
proving the detector reds on two statements sharing a line; and the eleven
file-backed steps' inline splitter copies were de-duplicated onto
`_split_sql_statements` via `_migration_file_statements`, preserving both
monkeypatch seams (`_execute_citation_migration_statement`,
`_execute_character_authority_migration_statement`) and V36→V37's
already-has-the-column skip. Verified a no-op, not assumed: the byte-identity
oracle re-run after the de-duplication is still 0 divergences against the
golden taken before any edit, and `Tests/ChaChaNotesDB/` + `Tests/DB/` is
1344 passed / 12 failed — the same 12 pre-existing failures the merge base has.
The only intentional behaviour delta is that a malformed migration file is now
rejected before any statement executes rather than after; both paths roll back
inside the step's transaction.

**Scoped out, deliberately.** 15 of the `DB/migrations/*.sql` files are
decorative twins of the embedded constants (never opened by any code path);
this port kept the embedded constant as the single source of truth for each
ported step rather than switching to the on-disk file, because switching would
have risked exactly the semantic divergence the byte-identity oracle exists to
prevent. The decorative-file cleanup remains its own task.

**Modified/added files:** `tldw_chatbook/DB/ChaChaNotes_DB.py`,
`Tests/ChaChaNotesDB/test_migration_atomicity.py` (new),
`Tests/ChaChaNotesDB/legacy_conversation_schema.py`,
`Tests/DB/test_chachanotes_citation_provenance_migration.py`,
`backlog/docs/lessons-testing-evidence.md`.
