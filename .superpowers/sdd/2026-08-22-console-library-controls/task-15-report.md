# Task 15 implementation report

## Outcome

Task 15 adds deterministic device-local dispatch recovery, exact explicit Retry
and Discard actions, restart-safe queue ownership, and the app-runtime-owned
ephemeral analogue. Restore reconciles a persisted conversation before any
queue consumer can wake. A valid local `accepted` or `dispatch_started`
checkpoint owns recovery; a checkpoint-free synchronized/imported state is
visible but inert; invalid, missing, cross-conversation, wrong-role, deleted, or
stale-version ownership is quarantined with a bounded code and no provider call.

An already-valid provider continuation remains authoritative. If continuation
and checkpoint ownership coexist, reconciliation moves the assistant to
`continuation_active` and deletes the exact checkpoint in one transaction; a
failure rolls both changes back. Task 15 does not create a tool-batch handoff,
normalize a new continuation, execute tools, or add history/sync/export
projections owned by Task 16.

Explicit durable Retry reuses the exact accepted USER and assistant, revalidates
the frozen Library authority and credential-free destination, reconstructs only
permitted inputs, and performs checkpoint CAS before provider entry.
`dispatch_started` is never replayed automatically and offers the literal
duplicate-risk warning. Retry is disabled with the exact model-owned reason when
one-shot prefill or transient evidence cannot be reconstructed. Repeated
in-flight actions are disabled and idempotent.

Provider completion, failure, cancellation, and Discard use the Task-6
repository's expected-revision settlement transaction. Assistant terminal
content/state/status/metadata/usage, USER and assistant version/deleted guards,
hash/sync intent, and exact checkpoint deletion commit together or all roll
back. A settlement fault restores the preterminal app-runtime assistant
projection, retains the same recovery owner and actions, publishes a blocked
nonterminal state, and prevents legacy terminal persistence, volatile cleanup,
or queue advancement.

Queued accepted work stays retired after restart, while later work remains
paused until exact Retry/Discard settlement; settlement releases the recovered
entry and drains later work exactly once. Ephemeral recovery lives only in the
store, survives screen/controller replacement, never writes a checkpoint row,
and blocks promotion with the exact literal `Finish or discard the pending turn
before saving.` Textual projects the UI-neutral model with literal text and
markup disabled.

## Governance and scope

The Task 15 Implementation Plan was added to TASK-19900.3 before production
edits with:

```text
ADR required: no.
ADR path: backlog/decisions/079-console-library-conversation-authority.md.
Reason: Task 15 directly implements ADR-079's approved recovery, terminal
settlement, queue-restart, and ephemeral ownership contracts without changing
the schema, persistence authority, or Task 16 continuation-handoff boundary.
```

TASK-19900.3 remains In Progress and all 22 acceptance criteria remain
unchecked. No schema, migration, table, transaction owner, persistence fallback,
second USER/assistant, or request-body checkpoint was added. No profile database
was opened, the app was not launched, no provider was contacted, and no full
repository sweep was run.

## Strict TDD evidence

All three Task 15 test files were complete before the first production edit:

- `Tests/Chat/test_console_dispatch_recovery.py`
- `Tests/Chat/test_console_dispatch_queue_recovery.py`
- `Tests/UI/test_console_dispatch_recovery.py`

Exact initial RED command:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_dispatch_recovery.py Tests/Chat/test_console_dispatch_queue_recovery.py Tests/UI/test_console_dispatch_recovery.py
```

Exact RED result:

```text
45 failed, 1 passed, 1 warning in 1.53s
```

Every failure was an intended missing production recovery seam or projection;
there were no fixture, collection, setup, or environment failures.

The atomic terminal failure barrier was then strengthened RED-first. Exact
command and result:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_dispatch_recovery.py -k 'retry_terminal_delete_failure or retry_cancel_settlement_failure'
3 failed, 31 deselected, 1 warning in 0.76s
```

Those failures proved that success/failure volatile assistant state could leak
past a rolled-back checkpoint delete and that cancellation propagated the raw
settlement error. After the dedicated settlement barrier and projection restore,
the same command passed all 3 selected tests.

A final run-state ratchet was also authored RED-first:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_dispatch_recovery.py -k 'discard_delete_failure or retry_terminal_delete_failure or retry_cancel_settlement_failure'
4 failed, 30 deselected, 1 warning in 0.82s
```

The DB rollback assertions already passed, but the runtime status was `IDLE` or
`STREAMING` instead of nonterminal `BLOCKED`. After the minimal run-state fix,
the same command produced `4 passed, 30 deselected, 1 warning in 0.76s`.

Exact fresh focused GREEN after all mutants were restored and all production
edits were complete:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_dispatch_recovery.py Tests/Chat/test_console_dispatch_queue_recovery.py Tests/UI/test_console_dispatch_recovery.py
53 passed, 1 warning in 6.77s
```

The focused matrix includes real temporary SQLite restart/reconciliation,
transaction failure, deleted/version-changed owner, continuation precedence,
remote/import inertness, provider ordering, provider success/failure/cancel,
Discard, queue restart/settlement, ephemeral lifetime/promotion, and Textual
literal-projection tests.

## Mutation evidence

Each mutant was applied independently, run against its named ratchet, killed,
and restored before the fresh focused GREEN:

1. Continuation precedence forced off: the both-owner authoritative-continuation
   test failed (1 failure).
2. Current USER version guard removed from reconciliation: the invalid-pair
   quarantine test failed (1 failure).
3. Assistant deleted/current-version guard removed from expected-revision
   settlement: the changed/deleted-owner settlement ratchet failed (1 failure).
4. Pre-provider checkpoint transition bypassed: the Retry ordering test failed
   (1 failure; provider observed `accepted` revision 1 instead of
   `dispatch_started` revision 2).
5. Exact checkpoint DELETE changed to SELECT and its row-count check disabled:
   the Discard atomicity test failed (1 failure because Discard falsely
   succeeded while retaining the checkpoint).
6. Queue hydration/wake recovery guard disabled: the restart wake test failed
   (1 failure; queue returned `APPLIED` instead of refusing advancement).
7. Automatic replay refusal disabled: the Task-14 provider-entry ratchet failed
   (1 failure; resume dispatched instead of returning Retry/Discard recovery).
8. Reconstructability forced true: both prefill/evidence parameter cases failed
   (2 failures). Replacing the exact disabled reason also failed both literal
   cases (2 failures).
9. Ephemeral promotion block removed: the promotion test failed (1 failure;
   execution entered persistence instead of returning the exact block copy).

The later atomic-settlement fault tests add three more killed failure boundaries
(provider success, provider failure, cancellation) plus the Discard rollback
boundary. No mutation was left in the final tree.

## Fresh adjacent verification

Task 14, including all four fix-round files and the first-send atomicity file:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_durable_turn_acceptance.py Tests/Chat/test_console_durable_turn_fix_round1.py Tests/Chat/test_console_durable_turn_fix_round2.py Tests/Chat/test_console_durable_turn_fix_round3.py Tests/Chat/test_console_durable_turn_fix_round4.py Tests/Chat/test_console_first_send_atomicity.py
73 passed, 1 warning in 31.79s
```

Exact Task 13 affected gate:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Architecture/test_console_wave6_inventory.py Tests/Chat/test_console_automatic_library_preparation.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_turn_library_authority.py Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_turn_preparation.py Tests/Chat/test_library_preparation.py Tests/UI/test_console_auto_rag_on_send.py Tests/UI/test_console_harness_config_honesty.py Tests/UI/test_console_rag_settings_modal.py Tests/UI/test_console_retrieval_controller.py Tests/UI/test_console_controller_wiring.py Tests/test_config_console_defaults.py
628 passed, 1 warning in 31.37s
```

Queue/runtime/UI companions:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_prompt_queue.py Tests/UI/test_console_prompt_queue.py Tests/UI/test_console_prompt_queue_modal.py Tests/UI/test_console_dispatch_recovery.py
95 passed, 1 warning in 12.13s
```

Repository/transaction/model/hydration/runtime-state companions:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py Tests/Chat/test_console_transaction_contribution.py Tests/Chat/test_console_chat_models.py Tests/Chat/test_console_conversation_hydration.py Tests/CI/test_textual_runtime_contract.py
126 passed, 1 warning in 17.84s
```

The controller-only adjacent run also passed 188 tests in 5.33s. Two exploratory
aggregate commands named nonexistent stale test paths and exited during
collection with code 4; they are not counted as evidence and were immediately
replaced by the exact repository paths shown above. No product test failed.

## Static, format, source, and privacy qualification

Scoped Ruff lint over all 17 changed/new Python production and test files:

```text
All checks passed!
```

Ruff format check over the 10 clean changed/new files:

```text
10 files already formatted
```

Six inherited whole-file formatter drifts remain in
`console_chat_models.py`, `console_chat_store.py`,
`console_dispatch_checkpoint.py`, `console_dispatch_repository.py`,
`UI/Console_Modules/prompt_queue.py`, and
`test_console_durable_turn_fix_round1.py`. Formatter diffs were inspected: Task
15's changed ranges are formatter-consistent; applying whole-file formatting
would retain large unrelated churn. `git diff --check` passed.

Privacy/log/source scans found one new fixed-name warning only:
`console_dispatch_recovery_hydration_failed`. There is no exception text,
request body, credential, API key, provider secret, prompt/evidence body, or
attachment content in the checkpoint or new logs. The only new-test
`authorization` match is the existing typed queue authorization fixture. No
migration/schema/SQL file changed. Retry reconstruction necessarily builds
provider messages and a Library query in runtime memory; neither is checkpointed
or logged.

All completed pytest gates emitted only the inherited environment-level
`RequestsDependencyWarning` for the installed requests/urllib3/charset package
versions. The warning predates Task 15 and is not hidden or changed here.

## Binding ruling and cost if wrong

Ruling: Task 15 intentionally replaces Task 14's explicitly transitional
post-terminal `dispatch_started` checkpoint assertions. Only tests/fakes that
encoded that deferred terminal seam were updated: terminal success, failure,
cancellation, and Discard now use atomic settlement and delete the checkpoint;
durable doubles explicitly implement `settle_with_assistant`. Production has no
compatibility fallback. Task 14's acceptance, exact-owner, fingerprint,
pre-provider CAS, and preterminal checkpoint assertions remain intact.

Cost if wrong: reverting this ruling requires restoring Task 14's transitional
terminal tests/fakes. Retaining the old seam after Task 15 would strand local
checkpoints after terminal publication, expose false restart recovery, and allow
queue work to remain paused despite a terminal assistant; adding a fallback
would instead weaken the mandatory atomic repository boundary.

## Files changed

Production:

- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Chat/console_chat_models.py`
- `tldw_chatbook/Chat/console_chat_store.py`
- `tldw_chatbook/Chat/console_dispatch_checkpoint.py`
- `tldw_chatbook/Chat/console_dispatch_repository.py`
- `tldw_chatbook/Chat/console_prompt_queue_coordinator.py`
- `tldw_chatbook/UI/Console_Modules/prompt_queue.py`
- `tldw_chatbook/UI/Console_Modules/dispatch_recovery.py`

Tests/fakes:

- `Tests/Chat/test_console_dispatch_recovery.py`
- `Tests/Chat/test_console_dispatch_queue_recovery.py`
- `Tests/UI/test_console_dispatch_recovery.py`
- `Tests/Chat/test_console_chat_controller.py`
- `Tests/Chat/test_console_durable_turn_fix_round1.py`
- `Tests/Chat/test_console_durable_turn_fix_round2.py`
- `Tests/Chat/test_console_first_send_atomicity.py`
- `Tests/Chat/test_console_prompt_queue_coordinator.py`

Governance/evidence:

- TASK-19900.3 Implementation Plan and Implementation Notes
- this Task 15 report
- shared progress ledger

## Self-review

- Verified reconciliation precedence and every exact delete/update CAS guard.
- Verified the queue hydration callback only advances a lifecycle revision and
  cannot recursively call activity/hydration.
- Verified recovery ownership and volatile assistant baselines are store-owned,
  bounded to the live session, and cleared only after terminal settlement or
  session/app-runtime disposal.
- Verified no terminal settlement failure reaches legacy `mark_failed`, system
  error publication, terminal persistence, continuation retirement, recovered
  queue settlement, or later queue advancement.
- Verified Retry revalidates authority/destination and crosses checkpoint CAS
  before the direct/agent provider seam while reusing exact message owners.
- Verified Task 16 boundaries remain unimplemented and Task 14 acceptance and
  fingerprint contracts remain authoritative.
- Reviewed the complete diff, untracked additions, status, whitespace, scoped
  static output, source/privacy scan output, and every final test result.

No unresolved implementation concern remains. The inherited dependency warning
and six baseline whole-file format drifts are explicitly qualified; neither was
expanded in this task.
