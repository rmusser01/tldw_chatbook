# Task 6 implementation report

## Scope and architecture

- Implemented only Task 6 from the frozen Console Library controls plan.
- Existing ADR applies: `backlog/decisions/079-console-library-conversation-authority.md`.
  No new architectural decision was introduced, so no new ADR was required.
- All database tests used pytest temporary file databases; no profile database was
  opened or mutated.

## RED evidence

The four required test files were created before production implementation and run
together with:

```text
../../.venv/bin/python -m pytest \
  Tests/ChaChaNotesDB/test_console_library_policy_repository.py \
  Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py \
  Tests/Chat/test_console_library_policy_coordinator.py \
  Tests/Chat/test_console_transaction_contribution.py -q
```

The intended RED was captured as four collection errors, one per file, because the
new repository/coordinator/checkpoint/contribution modules did not yet exist. An
initial attempt with bare `python` also established that this shell has no `python`
executable; all authoritative runs therefore used the repository virtualenv above.

Additional focused RED-to-GREEN cycles caught two hardening cases:

- `test_continuation_handoff_rejects_an_owner_that_has_not_started_dispatch`:
  initially failed because handoff accepted an `accepted` owner; handoff now requires
  matching `dispatch_started` ownership.
- `test_insert_integrity_failure_without_a_race_winner_is_unavailable`: initially
  failed because every integrity error was treated as a race; an integrity failure is
  now a typed unavailable result unless a durable winner can be read.
- The credential-query parameter of
  `test_checkpoint_codecs_reject_request_text_source_snippets_and_credentials`:
  initially failed because URL userinfo was rejected but query/fragment material was
  not; stored endpoints now reject all query and fragment data.

## GREEN evidence

Final required focused run:

```text
49 passed, 1 warning in 6.19s
```

The warning is the environment's existing `requests` dependency-version warning.

Narrowly adjacent compatibility run:

```text
Tests/Chat/test_console_library_policy.py
Tests/Chat/test_assistant_generation_state.py
Tests/Chat/test_chat_persistence_service.py
Tests/DB/test_chachanotes_console_library_policy_migration.py

118 passed, 1 warning in 14.16s
```

No full-suite sweep was run, per repository guidance and the Task 6 brief.

## Mutation evidence

Each mutation was applied separately, its named test was run with `-B` and
`PYTHONDONTWRITEBYTECODE=1`, the expected failure was observed, and the production
implementation was restored before the next probe:

1. Missing policy was mutated to return Automatic/Allowed.
   `test_read_distinguishes_valid_absent_corrupt_and_error_outcomes` failed at the
   absent-policy assertion: `1 failed` (expected).
2. Policy database error was mutated to return Automatic/Allowed.
   `test_read_distinguishes_valid_absent_corrupt_and_error_outcomes` failed at the
   error-result assertion: `1 failed` (expected).
3. The checkpoint revision predicate was removed from both the state-CAS preflight
   and conditional SQL update.
   `test_state_cas_requires_every_expected_owner_predicate[checkpoint_revision]`
   failed because the mutant committed: `1 failed` (expected).

The final 49-test GREEN run after restoration proves no mutation remained.

## Static and self-review evidence

- Scoped Ruff over all ten Task 6 production/test files: `All checks passed!`
- `git diff --check`: passed with no output.
- Reviewed every Task 6 SQL call: values are parameterized; there is no generic
  upsert or replacement statement.
- Reviewed transaction ownership: acceptance contributions receive only the active
  caller cursor, conversation ID, and user/assistant ID map; exceptions propagate.
- Reviewed atomic write boundaries: USER, assistant, checkpoint, state/message CAS,
  terminal update, sync trigger, checkpoint delete, continuation update, handoff
  delete, and contribution failures all have rollback coverage.
- Reviewed checkpoint privacy: exact canonical JSON shapes and 4096/2048/2048-byte
  caps are enforced; draft/request text, source snippets, credentials, provider
  request material, query strings, and fragments are rejected or absent.
- Reviewed recovery ownership: role, conversation, versions, deletion state,
  assistant state, active continuation absence, and one-owner cardinality fail closed.
- Reviewed coordinator ordering: DB work uses `asyncio.to_thread`; only committed
  writes publish in-process, and execution always performs a fresh durable read.

## Changed files

- `.superpowers/sdd/2026-08-22-console-library-controls/task-6-report.md`
- `tldw_chatbook/Chat/console_library_policy_repository.py`
- `tldw_chatbook/Chat/console_library_policy_coordinator.py`
- `tldw_chatbook/Chat/console_dispatch_checkpoint.py`
- `tldw_chatbook/Chat/console_dispatch_repository.py`
- `tldw_chatbook/Chat/console_transaction_contribution.py`
- `tldw_chatbook/Chat/chat_persistence_service.py`
- `Tests/ChaChaNotesDB/test_console_library_policy_repository.py`
- `Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py`
- `Tests/Chat/test_console_library_policy_coordinator.py`
- `Tests/Chat/test_console_transaction_contribution.py`

## Fix round 3 — exact transaction-writer INSERT grammar

### Finding verification and contract clarification

The review finding reproduced against fix-round-2 source. With a temporary in-memory
SQLite database, both
`INSERT INTO probe(value) VALUES ('literal ?')` with an empty parameter tuple and
`INSERT ... ON CONFLICT(value) DO UPDATE ...` executed through the writer. The old
validator looked only for an `INSERT` first token, any `?` character, and absence of
a semicolon. It therefore confused question marks in SQL text with bind parameters
and treated every SQLite INSERT extension as part of the capability.

The intended narrow capability is now stated consistently in the design spec,
frozen plan/ledger, ADR-079, and Task 6 brief: one statement of the exact form
`INSERT INTO simple_table (simple_column, ...) VALUES (?, ...)`, with ordinary
whitespace, unquoted and unqualified ASCII identifiers, one VALUES row, and equal non-zero
column, placeholder, and parameter arity. `executemany` requires at least one tuple
and matching arity for every row. This clarifies the already-approved insert-only
boundary; it does not expand the public writer interface.

### RED and GREEN evidence

All new regressions were written before the production validator changed. The
contribution-file RED run completed with:

```text
22 failed, 19 passed, 1 warning
```

The 22 failures covered literal and line/block-comment question marks; missing
column lists; INSERT OR REPLACE/IGNORE; ON CONFLICT DO UPDATE/DO NOTHING;
INSERT...SELECT; RETURNING; multiple VALUES rows; quoted/qualified identifiers;
column/placeholder/execute-tuple arity mismatches; and empty or wrong-arity
`executemany` rows. Existing transaction-control and non-INSERT rejections remained
green, and the wished-for ordinary-whitespace canonical INSERT already succeeded.

The first corrected contribution-file run completed with:

```text
41 passed, 1 warning
```

Self-review then exposed Python regular-expression Unicode case-folding: without an
ASCII flag, the nominal `[A-Z]` identifier range admitted `ſ`. The added regression
failed as intended with `1 failed, 24 passed`; adding `re.ASCII` made all 25 statement
cases pass. The final contribution file therefore contains 42 passing tests.

The four-file Task 6 focused suite then completed with:

```text
109 passed, 1 warning in 15.99s
```

The unchanged adjacent suite completed with:

```text
118 passed, 1 warning in 14.76s
```

The warning is the environment's existing `requests` dependency-version warning.
All SQLite databases were in-memory or pytest temporary databases; the profile
database was never opened. No full suite was run.

### Mutation and control evidence

The exact implementation was replaced with the old permissive first-token/substring
heuristic and the real statement matrix was rerun with bytecode disabled. Fourteen
formerly executable noncanonical INSERT forms each failed with `DID NOT RAISE`,
including both reviewer escapes, both conflict actions, comments/literals, SELECT,
RETURNING, multi-row VALUES, and quoted/qualified identifiers:

```text
14 failed, 10 passed, 1 warning
```

After restoration, the statement, execute-arity, executemany-arity, and raw
clear-authorizer/connection control cases completed together with:

```text
33 passed, 1 warning
```

The raw-cursor control continues to raise `AttributeError` before a contribution can
reach a connection or authorizer; this round did not reintroduce a cursor or SQLite
authorizer mechanism.

### Implementation and self-review

- A small anchored, case-insensitive standard-library regular expression accepts
  only the frozen grammar and ordinary whitespace/newlines. No SQL-parser dependency
  was added.
- The validator derives column and placeholder counts from the matched grammar.
  `execute` validates its exact tuple length; `executemany` materializes rows,
  rejects an empty batch, retains the tuple-only boundary, and validates every row
  before any write is delegated.
- The test tables make every SQL-valid prohibited INSERT form executable in real
  SQLite under the old heuristic, so those rejection cases do not pass merely
  because SQLite rejects malformed SQL. The separate Unicode case specifically
  proves that the validator rejects a non-ASCII identifier before SQLite lookup.
- Scoped Ruff over all ten Task 6 production/test files passed. Source proof found
  the anchored full-match and arity guards, consistent exact-grammar terminology in
  all four authorities, and no old first-token heuristic, authorizer, or raw-cursor
  contribution path. `git diff --check` passed.
- Final self-review found no grammar ambiguity or compatibility concern for the
  planned preparation/activity tables, whose table and column names are ordinary
  underscore identifiers and whose contributions need only single-row or batch
  placeholder VALUES inserts.

### Fix round 3 changed files

- `.superpowers/sdd/2026-08-22-console-library-controls/task-6-brief.md`
- `.superpowers/sdd/2026-08-22-console-library-controls/task-6-report.md`
- `Docs/superpowers/plans/2026-08-22-console-library-controls.md`
- `Docs/superpowers/specs/2026-08-22-console-library-controls-design.md`
- `backlog/decisions/079-console-library-conversation-authority.md`
- `tldw_chatbook/Chat/console_transaction_contribution.py`
- `Tests/Chat/test_console_transaction_contribution.py`

## Fix round 2 — transaction-writer capability correction

### Reviewer escape verification and superseded interpretation

The remaining review finding was reproduced against fix-round-1 source before
production edits. A contribution used the supplied raw cursor to call
`cursor.connection.set_authorizer(None)`, inserted its sidecar, committed, and
returned. The repository's `connection.in_transaction` postcondition raised only
after the irreversible commit. The regression failed with two persisted message rows
instead of zero.

This evidence supersedes fix round 1's Finding 8 contract interpretation and its
authorizer-based enforcement claim. The raw cursor itself supplied the public
authorizer mutator, so the authorizer could not be a capability boundary. The
historical section below is retained as the evidence available during that round;
the final contract and implementation are the writer capability described here.

Before production edits, the frozen interface was corrected consistently in the
design specification, implementation plan/interface ledger (including the later
preparation and activity contributions), ADR-079, and Task 6 brief.

### Corrected capability design

`ConsoleTransactionContribution.write(...)` now receives a
`ConsoleTransactionWriter`, conversation ID, and message-ID map. It no longer
receives `sqlite3.Cursor`.

The writer exposes only:

- `execute(statement, parameters)` for one parameterized INSERT; and
- `executemany(statement, parameter_rows)` for parameterized INSERT rows.

It rejects empty, multi-statement, non-INSERT, and non-parameterized SQL before
delegating to the private cursor. There is no public cursor, connection, authorizer,
transaction/savepoint, ATTACH/DETACH, commit/rollback, connection-factory,
repository, session, or publication capability. A new scoped writer is supplied to
each contribution and revoked when the callback returns or raises, so retaining the
object cannot create a later transaction. Contribution exceptions continue to
propagate through the caller-owned `BEGIN IMMEDIATE` transaction.

This is an application API capability boundary for trusted in-process components,
not a hostile-Python sandbox. Python reflection, arbitrary imports, and unrelated
global/process side effects remain outside this protocol's claim. The correction
removes the temporary SQLite authorizer rather than representing it as a security
boundary it cannot provide.

### RED evidence

RED was captured in two layers before production implementation:

1. The exact old-boundary exploit ran against the raw cursor implementation and
   failed at the rollback assertion because `2` message rows remained committed.
2. After the complete wished-for writer suite was authored, the combined four-file
   run stopped at the intended collection error:

   ```text
   ImportError: cannot import name 'ConsoleTransactionWriter'
   1 error, 1 warning
   ```

The new tests cover the public capability shape, callback-scope revocation,
single-row and batch parameterized INSERTs, the clear-authorizer/commit exploit,
direct commit/rollback/BEGIN/savepoint/release/ATTACH/DETACH attempts,
UPDATE/DELETE, unparameterized INSERT, multi-statement INSERT+COMMIT, and rollback
when a later contribution raises after an earlier contribution wrote successfully.

### GREEN and adjacent evidence

The corrected contribution file completed with:

```text
19 passed, 1 warning in 3.43s
```

The unchanged authoritative four-file Task 6 command completed with:

```text
86 passed, 1 warning in 10.95s
```

The narrow adjacent compatibility suite remained green:

```text
118 passed, 1 warning in 13.75s
```

The warning is the environment's existing `requests` dependency-version warning.
No full suite was run. Every database was a pytest temporary database; the profile
database was never opened.

### Mutation evidence

The 11 prior mutation probes remain recorded below at the exact fix-round-1 base;
their policy, codec, ownership, attachment, coordinator, and dispatch-CAS guards
were unchanged in this correction. A targeted mutation subset covered every altered
writer-boundary property. Each mutant was applied alone with bytecode disabled,
killed by the named real-behavior test, and restored before the final GREEN run:

1. The repository leaked its raw cursor as the `writer`; the
   clear-authorizer-and-commit case failed because it no longer raised and committed
   through the leaked connection.
2. `execute(...)` skipped statement validation; all 11 direct control/non-INSERT
   cases failed, including transaction, savepoint, ATTACH/DETACH, UPDATE/DELETE,
   unparameterized INSERT, and multi-statement SQL.
3. Callback-finally revocation was removed;
   `test_generic_contribution_receives_only_writer_and_committed_id_map` failed
   because the retained writer remained usable after the callback.
4. The repository swallowed a later contribution exception;
   `test_later_contribution_failure_rolls_back_normal_parameterized_writes` failed
   because no error propagated and the transaction committed.

### Static, documentation, and self-review evidence

- Scoped Ruff over all ten Task 6 production/test files: `All checks passed!`
- `git diff --check`: passed with no output.
- Documentation terminology check found `ConsoleTransactionWriter` and the same
  capability-boundary language in the spec, plan/frozen ledger, ADR-079, and Task 6
  brief. Both planned later contribution interfaces use `writer`, not `cursor`.
- The only remaining documented `cursor: sqlite3.Cursor` contribution-adjacent
  signature is `ConsoleDispatchRepository.insert_with_messages(...)`, whose cursor is
  private to the transaction-owning repository path and is not supplied to a
  contribution.
- Production source contains no `set_authorizer`, SQLite authorizer action constant,
  `writer=cursor`, or contribution `cursor=cursor` path.
- The public writer protocol exposes exactly two write methods and has no public
  connection-bearing state. The private implementation is slot-backed, validates
  before delegation, and revokes in `finally`.
- Normal insert-one/insert-many behavior, exception propagation, later-failure
  rollback, core message/checkpoint rollback, and no post-callback reuse all execute
  against real SQLite temporary databases.
- Final diff review found no unresolved correctness concern. Insert-only DML is the
  minimum needed by the current preparation/activity sidecars while preserving the
  generic contribution protocol for later planned implementations.

### Fix round 2 changed files

- `.superpowers/sdd/2026-08-22-console-library-controls/task-6-brief.md`
- `.superpowers/sdd/2026-08-22-console-library-controls/task-6-report.md`
- `Docs/superpowers/plans/2026-08-22-console-library-controls.md`
- `Docs/superpowers/specs/2026-08-22-console-library-controls-design.md`
- `backlog/decisions/079-console-library-conversation-authority.md`
- `tldw_chatbook/Chat/console_transaction_contribution.py`
- `tldw_chatbook/Chat/console_dispatch_repository.py`
- `Tests/Chat/test_console_transaction_contribution.py`


## Fix round 1 — Important review findings

### Finding verification and contract interpretation

All nine findings were checked against the v45 schema, existing message-attachment
storage, active-leaf behavior, transaction implementation, frozen interface ledger,
design specification, and ADR-079 before production edits.

- Findings 1–7 and 9 were valid as stated. Policy CAS/read validation did not cover
  every durable-row invariant, active recovery was conversation-wide instead of
  active-lineage-only, dispatch operations did not treat a soft-deleted conversation
  as inert, accepted attachments were discarded, checkpoint values were too
  permissive, identity relationships were incomplete, and execution capture could
  return a snapshot for a superseded binding.
- Finding 8 exposed a real escape through `sqlite3.Cursor.connection.commit()`.
  The frozen public protocol explicitly requires `sqlite3.Cursor`, so replacing it
  with a cursor-like wrapper would violate the consumer contract. The implementation
  instead installs a temporary SQLite authorizer while each contribution runs. It
  denies transaction control, savepoints, ATTACH, and DETACH on the supplied
  connection, restores the connection immediately afterward, and verifies that the
  caller transaction remains active. This preserves the exact protocol signature and
  actual cursor behavior. A Python protocol cannot sandbox arbitrary imported/global
  code or prevent code from independently calling `sqlite3.connect`; the enforceable
  boundary is therefore the supplied cursor/connection and the capabilities passed by
  this API. No repository, holder, session, publication callback, or connection
  factory is supplied.
- Accepted checkpoint `attempt_id` must equal the frozen authority attempt. A later
  `cas_state(..., new_attempt_id=...)` intentionally changes the operational attempt,
  so hydration enforces equality while the checkpoint is `accepted`; a valid
  `dispatch_started` checkpoint may carry the explicitly CASed retry attempt.
- Position zero of the accepted attachment set uses the established
  `messages.image_data`/`image_mime_type` storage, while positions at or above one use
  `message_attachments`. Both are written through the caller's transaction. The
  local-only active-leaf pointer is set to the accepted assistant owner in that same
  transaction so active-lineage recovery has a committed selected leaf.

### RED evidence

All regression tests were added to the same four Task 6 test files before production
edits. The combined run collected 72 tests and produced the intended result:

```text
24 failed, 48 passed
```

The 24 failures mapped to the review findings as follows:

- strict policy-current-row CAS and soft-deleted-policy read: 2;
- authority/reconstructability codec invariants, category restriction, identifier,
  error-code, and opaque-reference validation: 6;
- selected active-lineage ownership: 1;
- acceptance attempt relationship and malformed hydrated identities: 5;
- soft-deleted dispatch read/CAS/settlement/handoff: 4;
- attachment persistence, rollback, and reconstructability: 3;
- capture across a concurrent session rebind: 1;
- malicious contribution COMMIT and ATTACH attempts: 2.

### GREEN and adjacent evidence

The first complete GREEN attempt after implementation was:

```text
72 passed, 1 warning
```

After mutation restoration, strict post-commit policy validation, and self-review, the
unchanged final four-file command completed with:

```text
72 passed, 1 warning in 8.85s
```

The narrowly adjacent compatibility command remained green:

```text
Tests/Chat/test_console_library_policy.py
Tests/Chat/test_assistant_generation_state.py
Tests/Chat/test_chat_persistence_service.py
Tests/DB/test_chachanotes_console_library_policy_migration.py

118 passed, 1 warning in 14.49s
```

The warning in both commands is the environment's existing `requests` dependency
version warning. No full-suite run was performed. Every database used by the tests
was a pytest temporary file database; no profile database was opened.

### Mutation evidence

Each mutant was applied alone, the named test was run with `-B` and
`PYTHONDONTWRITEBYTECODE=1`, the expected failure was observed, and the source was
restored before the next probe. The final 72-test GREEN run proves restoration.

Original Task 6 probes:

1. Missing policy was changed to Automatic/Allowed;
   `test_read_distinguishes_valid_absent_corrupt_and_error_outcomes` failed.
2. Database read error was changed to Automatic/Allowed;
   `test_read_distinguishes_valid_absent_corrupt_and_error_outcomes` failed at the
   read-error assertion.
3. Both checkpoint-revision CAS guards were removed;
   `test_state_cas_requires_every_expected_owner_predicate[checkpoint_revision]`
   failed because the mutant committed.

Fix-round probes:

1. Current policy `schema_version` validation and its SQL predicate were removed;
   `test_compare_and_swap_refuses_a_corrupt_current_policy_row` failed because the
   mutant committed.
2. The active-lineage join was replaced with conversation-wide checkpoint selection;
   `test_read_considers_only_checkpoint_owners_on_the_selected_active_lineage`
   failed with quarantine instead of the selected committed owner.
3. Extra attachment persistence was disabled;
   `test_acceptance_persists_the_full_user_attachment_set_atomically` failed.
4. Unavailable authority was allowed to carry Automatic/Allowed;
   `test_authority_codec_rejects_fail_open_and_free_form_allowed_fields` failed
   because the mutant no longer raised.
5. Both accepted-attempt relationship checks were removed;
   `test_acceptance_requires_the_checkpoint_and_frozen_authority_attempt_to_match`
   failed because the mutant accepted the mismatch.
6. Contribution execution bypassed the transaction guard;
   `test_contribution_cannot_escape_the_caller_owned_transaction[early_commit]`
   failed because the mutant committed early.
7. The coordinator's binding-generation comparison was weakened;
   `test_capture_retries_when_session_is_rebound_during_the_durable_read` failed by
   returning Automatic/Allowed authority from the old conversation.
8. The mutation-side soft-deleted-conversation predicate was removed;
   `test_soft_deleted_conversation_cannot_recover_or_mutate_dispatch_ownership[cas]`
   failed instead of returning the required conflict.

### Static checks and self-review

- Scoped Ruff over all ten Task 6 production/test files: `All checks passed!`
- `git diff --check`: passed with no output.
- Source probes reconfirmed exact checkpoint byte caps `4096/2048/2048`, no
  `INSERT OR`, `REPLACE INTO`, generic upsert, or `SELECT *` in the repositories.
- Policy reads and committed writes now use one strict row decoder, include
  `conversations.deleted`, and reject invalid post-write durable rows before commit.
- Recovery reads follow only the selected active leaf through parent links. Owner
  cardinality is computed on that lineage; an inactive branch cannot hydrate or
  quarantine the selected branch.
- Dispatch reads and all mutation preflights include conversation deletion, exact
  checkpoint revision, exact stored/current USER and assistant message versions,
  owner roles/conversation IDs, `deleted = 0`, and matching assistant state.
- Attachment sidecar failure, contribution failure/escape, message/checkpoint CAS,
  settlement, sync-intent, checkpoint deletion, continuation write, and handoff
  deletion retain all-or-nothing rollback coverage.
- Checkpoint codecs retain exact key order and byte caps while enforcing policy
  source/revision/permission invariants, canonical source categories, bounded token
  identifiers, machine error codes, credential-free endpoints, and explicit
  `opaque:<token>` references. No draft, prefill, source snippet, credential, or
  provider request payload is persisted.
- Coordinator repository work remains off-loop. Publication occurs only after a
  committed write or a binding-validated fresh read; a superseded read is retried and
  repeated binding churn fails closed.
- No unresolved correctness concern remained after the final diff review. The
  protocol-level limitation around arbitrary independently imported side effects is
  explicit above rather than silently represented as process sandboxing.

### Fix round 1 changed files

- `.superpowers/sdd/2026-08-22-console-library-controls/task-6-report.md`
- `tldw_chatbook/Chat/console_library_policy_repository.py`
- `tldw_chatbook/Chat/console_library_policy_coordinator.py`
- `tldw_chatbook/Chat/console_dispatch_checkpoint.py`
- `tldw_chatbook/Chat/console_dispatch_repository.py`
- `Tests/ChaChaNotesDB/test_console_library_policy_repository.py`
- `Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py`
- `Tests/Chat/test_console_library_policy_coordinator.py`
- `Tests/Chat/test_console_transaction_contribution.py`
