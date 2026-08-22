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
