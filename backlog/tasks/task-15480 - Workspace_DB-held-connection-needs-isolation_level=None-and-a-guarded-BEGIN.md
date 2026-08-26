---
id: TASK-15480
title: Workspace_DB held connection needs isolation_level=None and a guarded BEGIN
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
labels:
  - bug
  - db
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the latency audit: the task-3011 held-connection port of Workspace_DB omitted the lesson task-3012 later documented in AgentRuns (`DB/AgentRuns_DB.py:77-85`): Python sqlite3's default isolation mode auto-BEGINs on DML, so any future bare DML issued through `connection()` accumulates an implicit transaction that makes an explicit `BEGIN` raise ("cannot start a transaction within a transaction") and silently rolls back on close. `DB/Workspace_DB.py:37-40` lacks `isolation_level=None`, and its `transaction()` (`:78-89`) runs a bare `BEGIN` without checking `in_transaction` (Research/Writing check it; Subscriptions documents its reasoning). Latent today — audit the call sites — but a correctness fuse for the next contributor. Also fix the stale comment at `Tools/workspace_file_roots.py:43-45` still describing Workspace_DB as fresh-connection-per-op.

Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace_DB holds its connection with isolation_level=None; the bare-DML call-site audit is recorded in the notes
- [x] #2 transaction() is safe when a transaction is already open (test)
- [x] #3 The stale workspace_file_roots comment is corrected; existing Workspace surfaces green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the task-3012 rationale in `DB/AgentRuns_DB.py:73-95` and the fresh
   task-15466 precedent in `DB/Library_Collections_DB.py` (its
   `_get_connection` autocommit comment and its `transaction()`/
   `read_transaction()` nesting-docstring wording) as the two templates to
   mirror.
2. Audit every call site that reaches `WorkspaceDB.connection()` or
   `.transaction()` (`Workspaces/registry_service.py`, plus
   `Workspace_DB.py`'s own `_initialize_schema`/`get_schema_version`) and
   classify each as read / DML-inside-`transaction()` / bare DML through
   `connection()`. Record the result before writing any fix.
3. TDD: add Workspace_DB cases to `Tests/DB/test_held_connections.py`
   (isolation_level is None; bare DML through `connection()` survives
   closing the held connection; an explicit `BEGIN` still works right
   after bare DML; nesting `transaction()` raises
   `sqlite3.OperationalError` and the outer block still rolls back
   cleanly) and confirm the autocommit-dependent ones are RED against
   current `dev`.
4. Add `conn.isolation_level = None` to `WorkspaceDB._get_connection`,
   with a comment mirroring the AgentRuns/Library_Collections rationale
   and citing this task's audit result.
5. Add the nesting paragraph to `transaction()`'s docstring, worded like
   `Library_Collections_DB.transaction()`'s.
6. Fix the stale "fresh connection per operation" comment at
   `Tools/workspace_file_roots.py:43-45`.
7. Confirm the new tests are GREEN, then run the existing Workspace_DB
   suite (`Tests/Workspaces/`, `Tests/Tools/test_*workspace_file_roots*`)
   and the wider Workspace UI/Agents/Notes consumer suites for
   regressions, plus the pragma-pairing regression test
   (`Tests/DB/test_pragma_settings.py`, which already parametrizes
   `WorkspaceDB` into `_HELD_CONNECTION_DBS`).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Call-site audit (AC#1)** — every reachable `WorkspaceDB.connection()` /
`.transaction()` use, by grep + manual read of `Workspaces/registry_service.py`
and `DB/Workspace_DB.py`:

- `Workspace_DB.py:114` (`_initialize_schema`'s `connection()` block): DDL +
  an explicit `conn.commit()` inside the block, self-committing under
  either isolation mode. Its v2-migration writes run inside
  `self.transaction()` (`:253`), not here.
- `Workspace_DB.py:272` (`get_schema_version`): pure read.
- `Workspaces/registry_service.py`: 11 `connection()` call sites, ALL pure
  `SELECT`s (`list_workspaces`, `get_workspace`, membership lookups,
  runtime-binding lookups, change-review/rag-scope reads, the default-
  workspace legacy-binding check). Every write (`create_workspace`,
  `rename_workspace`, archive/unarchive, `set_active_workspace`, membership
  insert, runtime-binding upsert/delete, change-review toggle, rag-scope
  upsert/delete, default-workspace reseed) goes through `self.db.transaction()`.
  One `transaction()` call (`:576`) wraps a pure `SELECT` — wasteful but not
  a correctness issue.

**Conclusion: no bare DML through `connection()` exists in production
today.** This confirms the task description's "latent, not live" framing —
the fix closes a correctness fuse for the next contributor rather than
fixing an active data-loss bug. Two new tests
(`test_bare_dml_through_connection_survives_closing_the_held_connection`,
`test_explicit_begin_still_works_after_bare_dml`) exercise the scenario
synthetically (direct `INSERT` through `db.connection()`, mirroring how
`Tests/DB/test_held_connections.py` already pins the same property for
`RAGIndexingDB`) since there is no real call site to exercise it through.

**Fix (AC#1, AC#2)** — `WorkspaceDB._get_connection` now sets
`conn.isolation_level = None` (mirrors `AgentRuns_DB`/`Library_Collections_DB`,
with a comment citing the audit above). `transaction()`'s existing
try/except-rollback-reraise shape (unchanged) already made a nested
`transaction()` call raise `sqlite3.OperationalError: cannot start a
transaction within a transaction` with the outer block rolling back
cleanly — the TDD run confirmed this specific test was already green pre-fix,
so AC#2 only needed the nesting-docstring paragraph (worded like
`Library_Collections_DB.transaction()`'s) and a pinning test, not a
behavior change. Went with the "loud raise, clean outer rollback" shape
(the `Library_Collections_DB`/`AgentRuns_DB` precedent) rather than the
`in_transaction`-guard shape (`Research_DB`/`Writing_DB`) since that is
what the two freshest (task-15466) ports already established and what
`WorkspaceDB.transaction()`'s existing try/except already provides for
free. Family split (per review round 1): the held-connection DBs
(`WorkspaceDB`, `Library_Collections_DB`, `AgentRunsDB`) all bare-`BEGIN`-
and-raise on nesting — no in-transaction guard, nesting is a bug the
caller must not do, and it fails loudly; `Research_DB`/`Writing_DB` are
the older per-call-connection family and instead JOIN an already-open
transaction via an `in_transaction` check. This task did not add a code
guard to `WorkspaceDB.transaction()` — it documents and pins the
already-correct raise-and-rollback behavior the held-connection family
shares.

**TDD verification**: added the four new tests first and ran them against
pre-fix `Workspace_DB.py` — `test_isolation_level_is_none`,
`test_bare_dml_through_connection_survives_closing_the_held_connection`,
and `test_explicit_begin_still_works_after_bare_dml` were RED (isolation_level
was `''` not `None`; the bare-DML row was lost on close; the immediate
follow-up write raised `OperationalError`); `test_nesting_a_transaction_raises_and_the_outer_block_rolls_back`
was already GREEN pre-fix, as expected. All four are GREEN after the fix.

**AC#3**: corrected the stale "`WorkspaceDB` opens a fresh `sqlite3`
connection per operation" claim in
`Tools/workspace_file_roots.py:43-47`'s `_default_registry_factory`
docstring to describe the actual task-3011 per-thread held-connection
shape. Review round 1 confirmed two more stale comments of the same
"fresh connection per call" claim and asked for both to be folded in
rather than deferred: `Chat/console_chat_controller.py:639` (a memoization
rationale comment — reworded to "a repeat query against `WorkspaceDB`'s
held, per-thread connection -- task-3011") and
`UI/Console_Modules/workspace.py:1132` (the `:memory:`-guard rationale in
`_read_console_workspace_scope` — reworded to explain the guard's actual
current reason: a per-thread held connection would hand each new thread
its own empty, table-less `:memory:` database, since `WorkspaceDB` — unlike
`ClientNotificationsDB` — has no cross-thread `:memory:`-sharing branch).
All three fixed comments now describe the real task-3011 held-connection
shape.

**Review round 1 (4 Minors, all fixed, mechanical)**:
1. The `TestWorkspaceDBAutocommitAndNesting` class insertion had silently
   dropped the pre-existing `db.close()` teardown from
   `TestNotificationSettingsAreWrittenAtomically.test_multi_key_update_uses_one_transaction`
   (an artifact of inserting the new section between it and the next
   class) and left a duplicated `db.close()` at the end of the new nesting
   test. Restored the missing teardown call, deleted the duplicate.
2. Corrected two more stale "`WorkspaceDB` opens a fresh/brand-new
   connection per call" comments the reviewer found, same family as the
   `workspace_file_roots.py` one this task already fixed:
   `Chat/console_chat_controller.py:639` and
   `UI/Console_Modules/workspace.py:1132` (details above, replacing the
   earlier "left untouched, follow-up task" note — both are now fixed in
   this task per reviewer instruction).
3. Fixed the "13 `connection()` call sites in `registry_service.py`" count
   above to the correct 11 (13 was double-counting `Workspace_DB.py`'s own
   2 sites, which are listed separately in the bullet above it).
4. The commit subject overstated the fix as "guarded BEGIN" — there is no
   `in_transaction` guard; nesting deliberately raises. Amended locally
   (unpushed) to `fix: autocommit + documented nesting semantics for
   Workspace_DB held connection (task-15480)`, and the family-split
   sentence above records the reviewer's point that held-connection DBs
   raise on nesting while the older per-call DBs join.

**Testing**: `Tests/DB/test_held_connections.py` (32 passed, including the
4 new/extended Workspace_DB cases and the round-1 teardown fix),
`Tests/DB/test_pragma_settings.py` + `Tests/Workspaces/` +
`Tests/Tools/test_file_tools_workspace_roots.py` +
`Tests/Tools/test_workspace_file_roots.py` (294 passed — `WorkspaceDB` was
already present in `_HELD_CONNECTION_DBS`, no change needed there),
`Tests/Chat/test_console_chat_controller.py` (172 passed — covers the
round-1 comment fix in that file), and the wider Workspace UI/Agents/Notes
consumer suites (`Tests/UI/test_console_workspace_*`,
`test_settings_workspaces_category.py`, `test_post_release_workspaces_library_depth.py`,
`Tests/Agents/test_builtin_provider_workspace_binding.py`,
`test_run_log_workspace_isolation.py`, `Tests/Notes/test_server_notes_workspace_service.py`,
150 passed — covers the round-1 comment fix in `UI/Console_Modules/workspace.py`).
`Tests/DB/` full-directory run showed the pre-existing dev baseline
failures (32, all ChaChaNotes V33/V34 migration + `test_sql_validation.py`
schema-drift tests) — unrelated to this change, matches the documented
known-baseline note.

**Files touched**: `tldw_chatbook/DB/Workspace_DB.py` (autocommit fix +
transaction nesting docstring), `tldw_chatbook/Tools/workspace_file_roots.py`
(stale comment fix), `tldw_chatbook/Chat/console_chat_controller.py` +
`tldw_chatbook/UI/Console_Modules/workspace.py` (round-1: two more stale
"fresh connection per call" comments), `Tests/DB/test_held_connections.py`
(new `TestWorkspaceDBAutocommitAndNesting` class, 4 tests, a shared
`_insert_workspace_record` helper, and the round-1 teardown fix).
<!-- SECTION:NOTES:END -->
