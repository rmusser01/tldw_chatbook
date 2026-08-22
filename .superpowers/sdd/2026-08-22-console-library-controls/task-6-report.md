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
