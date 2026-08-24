# Task 16 report — hand dispatch recovery to ADR-063 continuations

## Outcome

Task 16 is implemented on the approved Task 15 head
`fc583e8dc1a51979dcbe84767d49f490fb85faf1`. The first supported durable
tool-call batch now transfers the exact Task 14/15 assistant from device-local
dispatch ownership to ADR-063 continuation ownership in the existing SQLite
transaction before any tool observer can run. Legacy/imported active
continuations normalize lazily under exact version/deletion/active-path CAS,
and one shared closed-state predicate now governs ordinary provider history.
Portable state remains part of the existing whole-message projections while
device-local dispatch checkpoints remain absent.

TASK-19900.3 remains **In Progress** with all 22 acceptance criteria unchecked.
No Task 17+ work, schema change, profile database, app launch, network/provider
call, full repository test sweep, push, or new ADR was used.

ADR required: no.

ADR paths:

- `backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md`
- `backlog/decisions/079-console-library-conversation-authority.md`

Reason: this slice directly implements the two existing ADRs' exclusive
dispatch-to-continuation ownership, normalization, history, and projection
contracts without adding another owner, schema, protocol, or transaction
boundary.

## Clean-head and baseline evidence

The source checkout was pinned before test or production edits:

```text
git rev-parse HEAD
fc583e8dc1a51979dcbe84767d49f490fb85faf1

git status --short
<empty>
```

Import provenance used the repository's existing probe:

```text
../../.venv/bin/python -B -m pytest -q Tests/test_probe_import_provenance.py
1 passed
```

The requested pre-Task16 adjacent baseline was:

```text
../../.venv/bin/python -B -m pytest -q --tb=short --show-capture=no Tests/Chat/test_provider_continuation_crash_recovery.py Tests/Chat/test_provider_continuation_history.py Tests/Sync_Interop/test_chat_outbox_producer.py Tests/Sync_Interop/test_envelope_builder.py Tests/Sync_Interop/test_envelope_applier.py Tests/Sync_Interop/test_provider_continuation_reconciliation.py Tests/Chatbooks/test_provider_continuation_roundtrip.py Tests/Chat/test_assistant_generation_state_roundtrip.py Tests/Chat/test_assistant_generation_state.py
4 failed, 165 passed, 1 warning
```

All four baseline failures were meaningful continuation crash-path failures:
the historical continuation writer did not commit the newly required closed
state, so its whole-message local sync intent could not prove the continuation
payload. They were not collection, fixture, import, or temporary-database
failures.

The only warning throughout the pytest evidence was the inherited environment
warning from `requests` about the installed urllib3/chardet combination.

## Strict RED

The complete two-file wished-for matrix was authored before the first
production edit. It exercised real temporary SQLite repositories, the pure
agent runtime, AgentService wiring, the production Store callback, active-path
restore, Sync-v2 encryption/outbox state, and the actual continuation recovery
model.

```text
../../.venv/bin/python -B -m pytest -q --tb=short --show-capture=no Tests/Chat/test_provider_continuation_crash_recovery.py Tests/Chat/test_provider_continuation_history.py Tests/Chat/test_console_dispatch_continuation_handoff.py Tests/Chat/test_console_assistant_generation_history.py Tests/Sync_Interop/test_chat_outbox_producer.py Tests/Sync_Interop/test_envelope_builder.py Tests/Sync_Interop/test_envelope_applier.py Tests/Sync_Interop/test_provider_continuation_reconciliation.py Tests/Chatbooks/test_provider_continuation_roundtrip.py Tests/Chat/test_assistant_generation_state_roundtrip.py Tests/Chat/test_assistant_generation_state.py
67 failed, 177 passed, 1 warning in 20.84s
```

The 67 intended failures covered:

- no exclusive first-tool-batch handoff in the production Store callback;
- statement and SQLite COMMIT rollback, pre-tool ordering, exact owner/version,
  deletion, destination, hash, local intent, and Sync-v2 proof;
- Ruling A's two distinct failures: local transaction failure retained dispatch
  recovery, while post-local-commit Sync-v2 failure retained only the valid
  continuation and ran zero tools;
- reasoning-only terminal continuation settlement rather than active recovery;
- dual-owner precedence and checkpoint-free NULL/stale normalization;
- action-disable timing, committed version rebind, known rollback confirmation,
  same-newer-version retry, and changed/deleted/replaced/invalid fail-closed
  races;
- the full NULL/accepted/dispatch-started/continuation-active/complete/stopped/
  failed/discarded × empty/non-empty × active-continuation Cartesian provider
  history contract through the real Console builder;
- active JSON/screen-state/legacy Character history and bounded literal export
  projection.

No RED failure was caused by collection or an invalid setup.

## Implementation and contract decisions

### Exclusive handoff and Ruling A

`ConsoleChatStore.persist_provider_continuation_event` now detects the exact
live Task 15 dispatch owner for `ToolBatchReady`, freezes a
`ConsoleContinuationHandoff`, and calls the existing
`ConsoleDispatchRepository.handoff_to_provider_continuation` transaction. The
repository still owns message/checkpoint CAS; Task 16 added frozen resolved
destination matching and consumes the committed message version and canonical
payload hash returned by that transaction.

The pure agent runtime calls the persistence callback before review/invocation,
so a local UPDATE, DELETE, or SQLite COMMIT failure raises back through the loop,
runs zero tools, and restores the same runtime-inactive/actionable dispatch
owner. Once the local transaction commits, the Store publishes the continuation
as the sole owner and runs the required Sync-v2 projection barrier. Projection
failure raises before tools but never compensates, recreates a checkpoint, or
hands ownership back. This is the authoritative Ruling A boundary.

The accepted v2 sync intent may already have been projected during restore. The
handoff test therefore asserts the authoritative event sequence `[2, 3]` and
exactly one v3 continuation projection, rather than incorrectly treating the
accepted projection as a duplicate continuation.

`FinalContinuation` on the still-live dispatch owner uses the existing atomic
terminal settlement transaction. A complete reasoning-only continuation is
written with assistant content/state/hash/sync intent while the dispatch
checkpoint is deleted in the same transaction; it never becomes an active
continuation recovery owner.

### Lazy normalization and action proof

Restore continues to reconcile dispatch ownership before queue wake. A valid
active ADR-063 continuation wins over stale or NULL assistant state and over a
same-assistant stale dispatch checkpoint. Checkpoint-free legacy owners are
first hydrated with actions disabled, then freshly read and normalized under:

- exact conversation and active path;
- assistant role and `deleted = 0`;
- exact message version and prior state;
- exact canonical continuation identity.

A committed normalization rebinds the new version/hash before enabling actions.
A conflict re-reads and retries only the identical continuation at the newly
observed version. Changed, deleted, replaced, invalid, or off-path owners lose
their stale action handle. A known rolled-back write preserves actions only
after a fresh read proves the identical owner/version/state/continuation; the
visible warning remains bounded and contains no private continuation data.

### One history policy and projections

`assistant_state_allows_provider_history` is the one pure ordinary-history
policy. It excludes active continuation sidecar owners, accepted,
dispatch-started, continuation-active-without-valid-continuation, failed,
discarded, and every empty assistant row. It admits legacy NULL state and
complete/stopped content. Completed continuation history retains the existing
provider-target behavior: matching private history is grouped/budgeted through
the ADR-063 sidecar, while a provider switch may use its visible completed text
without leaking private reasoning.

The Console provider builder and legacy Character history helper use the shared
predicate. The Console message model, active conversation hydration, and
screen-state serialization now carry the portable closed state. Restored
continuation actions additionally require a freshly rebound committed handle;
the Textual callout disables both its buttons and callback while that proof is
missing. Existing Task 5 Sync-v1/v2, Chatbook, text/Markdown/document,
trajectory, and import projection contracts remain authoritative and passed
unchanged. `console_dispatch_checkpoints` remains absent from those projection
modules.

## GREEN evidence

First focused GREEN after the two remaining fixture/contract reconciliations:

```text
../../.venv/bin/python -B -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_dispatch_continuation_handoff.py Tests/Chat/test_console_assistant_generation_history.py
75 passed, 1 warning in 7.88s
```

The adjacent ADR-063 crash matrix was then migrated from its legacy
continuation-only setup to the approved Task 14/15 durable dispatch owner plus
portable Sync-v2 fixture. All seven crash locations and the runtime restart
side-effect ratchets remained intact:

```text
../../.venv/bin/python -B -m pytest -q --tb=short --show-capture=no Tests/Chat/test_provider_continuation_crash_recovery.py
12 passed, 1 warning in 3.14s
```

Fresh post-restoration final focused and adjacent gates:

```text
../../.venv/bin/python -B -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_dispatch_continuation_handoff.py Tests/Chat/test_console_assistant_generation_history.py
75 passed, 1 warning in 6.99s

../../.venv/bin/python -B -m pytest -q --tb=short --show-capture=no Tests/Chat/test_provider_continuation_crash_recovery.py Tests/Chat/test_provider_continuation_history.py Tests/Chat/test_console_dispatch_continuation_handoff.py Tests/Chat/test_console_assistant_generation_history.py Tests/Sync_Interop/test_chat_outbox_producer.py Tests/Sync_Interop/test_envelope_builder.py Tests/Sync_Interop/test_envelope_applier.py Tests/Sync_Interop/test_provider_continuation_reconciliation.py Tests/Chatbooks/test_provider_continuation_roundtrip.py Tests/Chat/test_assistant_generation_state_roundtrip.py Tests/Chat/test_assistant_generation_state.py
244 passed, 1 warning in 34.69s

../../.venv/bin/python -B -m pytest -q --tb=short --show-capture=no Tests/Architecture/test_console_wave6_inventory.py Tests/Chat/test_console_automatic_library_preparation.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_turn_library_authority.py Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_turn_preparation.py Tests/Chat/test_library_preparation.py Tests/UI/test_console_auto_rag_on_send.py Tests/UI/test_console_harness_config_honesty.py Tests/UI/test_console_rag_settings_modal.py Tests/UI/test_console_retrieval_controller.py Tests/UI/test_console_controller_wiring.py Tests/test_config_console_defaults.py
628 passed, 1 warning in 48.98s

../../.venv/bin/python -B -m pytest -q --tb=short --show-capture=no Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py Tests/Chat/test_console_transaction_contribution.py Tests/Chat/test_console_chat_models.py Tests/Chat/test_console_conversation_hydration.py Tests/CI/test_textual_runtime_contract.py
126 passed, 1 warning in 22.34s

../../.venv/bin/python -B -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_dispatch_recovery.py Tests/Chat/test_console_dispatch_queue_recovery.py Tests/UI/test_console_dispatch_recovery.py Tests/Chat/test_console_dispatch_recovery_fix_round1.py Tests/UI/test_console_dispatch_recovery_fix_round1.py Tests/Chat/test_console_dispatch_recovery_fix_round2.py Tests/UI/test_console_dispatch_recovery_fix_round2.py Tests/Chat/test_console_dispatch_recovery_fix_round3.py Tests/Chat/test_console_dispatch_recovery_fix_round4.py
101 passed, 1 warning in 26.74s

../../.venv/bin/python -B -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_durable_turn_acceptance.py Tests/Chat/test_console_durable_turn_fix_round1.py Tests/Chat/test_console_durable_turn_fix_round2.py Tests/Chat/test_console_durable_turn_fix_round3.py Tests/Chat/test_console_durable_turn_fix_round4.py Tests/Chat/test_console_first_send_atomicity.py
73 passed, 1 warning in 48.01s
```

The 628 gate initially exposed three completed-continuation history assertions.
Evidence showed the first predicate call treated every valid continuation as
active. The minimal correction passes `has_valid_continuation=True` only for an
active checkpoint: active owners remain sidecar-only, while completed
provider-matching history keeps existing private grouping and provider-switch
visible history. The focused three-test correction and final 628 rerun passed.

One extra, non-gating historical Textual send test timed out. The exact same
test also timed out from a clean archive of the approved `fc583e8d` parent with
the same hidden composer/retained draft state, proving it is not introduced by
Task 16. No app or profile database was opened to investigate that independent
baseline.

## Mutation evidence

Every mutant was applied alone and restored before the next one:

1. **Leave dispatch checkpoint after continuation write.**
   `test_first_tool_batch_uses_atomic_handoff_and_publishes_committed_proof`
   failed `checkpoint_count 1 != 0` (1 failed).
2. **Ignore the current durable USER version.**
   The `[user_version]` handoff guard failed because the mutation no longer
   raised (1 failed).
3. **Admit continuation-active/active continuation through ordinary history.**
   The Cartesian builder ratchet failed five cases (5 failed, 28 passed).
4. **Admit unresolved/failed/discarded or empty ordinary history.**
   The Cartesian builder ratchet failed eleven cases (11 failed, 21 passed).
5. **Omit `assistant_generation_state` from one Sync-v2 envelope surface.**
   All seven closed-state projection variants failed (7 failed).
6. **Expose Retry for checkpoint-free remote accepted state.**
   The inert source-device action assertion failed with one real enabled action
   (1 failed, 1 passed).

`rg -n "MUTATION PROBE"` over the mutated production files returned no matches
after restoration, and the fresh 244/628/126/101/73 gates above ran only after
that restoration.

## Static, privacy, and source checks

```text
../../.venv/bin/python -m ruff check <all changed Python production/test files>
All checks passed!

../../.venv/bin/python -m ruff format --check Tests/Chat/test_console_assistant_generation_history.py Tests/Chat/test_console_dispatch_continuation_handoff.py
2 files already formatted

git diff --check
<empty>

rg -n 'console_dispatch_checkpoints' tldw_chatbook/Sync_Interop tldw_chatbook/Chatbooks tldw_chatbook/Chat/trajectory_export.py tldw_chatbook/Character_Chat
<empty>

rg -n 'assistant_generation_state' tldw_chatbook/Sync_Interop tldw_chatbook/Chatbooks tldw_chatbook/Chat/trajectory_export.py tldw_chatbook/Character_Chat/local_character_persona_service.py | wc -l
43

git diff -U0 -- '*.py' | rg -n '^\\+.*logger.*(content|prompt|evidence|checkpoint|provider_continuation_json|api[_-]?key)'
<empty>
```

Whole-file Ruff formatting is not claimed for inherited large files: the
approved parent already fails repository-wide/whole-large-file formatting in
these modules. The two new test files are formatter-clean; changed production
ranges were reviewed manually, Ruff lint is clean, and `git diff --check`
passes.

No new log line includes prompt, evidence, continuation body/reasoning, tool
arguments/results, destination credentials, or provider secrets. UI warning
copy remains fixed/bounded. Tests use only generated keys and temporary SQLite.

## Files changed

- `Tests/Chat/test_console_dispatch_continuation_handoff.py`
- `Tests/Chat/test_console_assistant_generation_history.py`
- `Tests/Chat/test_provider_continuation_crash_recovery.py`
- `Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py`
- `tldw_chatbook/Chat/assistant_generation_state.py`
- `tldw_chatbook/Chat/console_chat_models.py`
- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Chat/console_chat_store.py`
- `tldw_chatbook/Chat/console_conversation_hydration.py`
- `tldw_chatbook/Chat/console_dispatch_checkpoint.py`
- `tldw_chatbook/Chat/console_dispatch_repository.py`
- `tldw_chatbook/DB/ChaChaNotes_DB.py`
- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py`
- `tldw_chatbook/UI/Console_Modules/message.py`
- `tldw_chatbook/UI/Console_Modules/provider_continuation_recovery.py`
- `backlog/tasks/task-19900.3 - Make-automatic-Console-Library-retrieval-a-truthful-send-gate.md`
- `.superpowers/sdd/2026-08-22-console-library-controls/progress.md`
- this report.

## Self-review

- Confirmed local transaction failure and post-local-commit Sync-v2 failure
  have deliberately different owners and neither path can run tools.
- Confirmed no second USER/assistant, persistence fallback, schema/table, or
  Task16-owned transaction was added.
- Confirmed handoff validates exact checkpoint/USER/assistant versions, roles,
  conversation/deleted state, canonical continuation, dispatch-started state,
  and credential-free frozen destination before ownership transfer.
- Confirmed terminal reasoning-only completion uses atomic assistant update plus
  checkpoint deletion and preserves the complete continuation for provider
  history/projection.
- Confirmed action proof is runtime-only and is not serialized as portable
  authority; navigation keeps the app-owned Store, while restart rebinds from
  SQLite.
- Confirmed active continuations never enter ordinary history, completed
  provider history remains target-aware, and the Character helper uses the same
  predicate.
- Confirmed all mutants were restored and no full suite/push/network/profile DB
  was used.

That was the pre-review conclusion. Independent review subsequently identified
five continuation-recovery and mounted-projection defects; the correction and
its qualified evidence follow below. The inherited Requests warning,
parent-baseline Textual timeout, and whole-large-file formatter baseline remain
qualified rather than claimed fixed.

## Independent review correction

### Verified findings and strict RED

The withdrawn stale-USER-version report was not implemented: exact USER
version/deleted quarantine remains unchanged. The five final findings were
reproduced through production repository/store/controller paths, temporary
SQLite, and mounted Textual widgets before production edits. The live in-flight
Markdown control was corrected to inspect the enclosing Assistant-turn header,
which is the actual mounted owner of that copy; it passed at RED and remains a
precedence control.

```text
../../.venv/bin/python -m pytest -q --tb=line -p no:logging Tests/Chat/test_console_continuation_review_fixes.py Tests/UI/test_console_continuation_review_fixes.py
12 failed, 1 passed, 1 warning in 2.81s
```

The twelve intended failures mapped to:

- three recursive active-path cases: a sole earlier valid continuation was
  missed, while duplicate and orphan claimants were not quarantined;
- one continuation-row read failure that removed the neutral recovery owner
  and allowed submission to fail open;
- two real full-controller reasoning-only terminal cases, manual and queued,
  where the Store's atomic `FinalContinuation` settlement was followed by a
  second controller terminal mutation;
- five mounted empty-state rows (`accepted`, `dispatch_started`, `complete`,
  `failed`, and `discarded`) that retained the state but rendered an empty
  Markdown body; and
- one ambiguous/executing continuation whose disabled action proof was dropped
  by the UI-neutral projection.

The original queued test admitted two queued entries. During GREEN it exposed
the established safety rule that durable queued acceptance pauses later work;
the ratchet was narrowed to one real claimed queued turn, the exact review
finding. The terminal-recognition mutant below proves that final ratchet fails
on the old double-write behavior rather than passing due to its fixture.

### Minimal correction

- Checkpoint-free repository reconciliation now scans the whole recursive
  active path. It accepts only one valid active continuation, quarantines
  duplicate/orphan/invalid claims with bounded codes, treats an exact
  complete-continuation/complete-assistant pair as inert terminal history, and
  uses the active leaf only for the existing inert remote
  accepted/dispatch-started fallback.
- A continuation hydration read failure replaces the transient continuation
  recovery with the same assistant/conversation identity in a bounded
  `continuation_hydration_error` quarantine. It has no actions, blocks Send,
  and survives until a new Store performs an exact successful re-read and
  normalization.
- Both plain and Markdown transcript bodies use
  `render_exported_assistant_content` after the existing live
  activity/generating handling. Stored content remains unchanged; the mounted
  literals are bounded and live in-flight copy still wins.
- The Store can prove an event-settled terminal message only when the in-memory
  and durable role/deletion/version/state/content/complete-continuation facts
  agree and no dispatch recovery remains. The controller consumes that exact
  snapshot before fallback or terminal mutation, so manual and queued
  reasoning-only completion write once.
- Ambiguous/executing continuation projection now carries the Store's action
  proof, keeping mounted Discard disabled when the recovered handle has not
  been freshly rebound.

Ruling A remains unchanged: local SQLite handoff failure retains dispatch
recovery, while post-local-commit Sync-v2 failure retains only the committed
ADR-063 continuation and runs zero tools.

### GREEN and adjacent evidence

First focused GREEN and the fresh post-mutation restoration run were:

```text
../../.venv/bin/python -m pytest -q --tb=short -p no:logging Tests/Chat/test_console_continuation_review_fixes.py Tests/UI/test_console_continuation_review_fixes.py
13 passed, 1 warning in 2.86s

../../.venv/bin/python -B -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_continuation_review_fixes.py Tests/UI/test_console_continuation_review_fixes.py
13 passed, 1 warning in 2.82s
```

Fresh targeted companion gates after every mutant was restored:

```text
# Task16/ADR-063/Task5 plus the new review matrix
258 passed, 1 warning in 24.05s

# Repository/state
126 passed, 1 warning in 19.70s

# Task15 recovery
101 passed, 1 warning in 22.74s

# Task14 durability
73 passed, 1 warning in 35.92s

# Exact Task13/controller gate
628 passed, 1 warning in 36.52s
```

One deliberately broader, non-gating UI companion finished `212 passed, 5
failed, 1 warning in 234.46s`. All five failures are stale assertions in
`Tests/UI/test_console_transcript_markdown_widget.py` that query the Markdown
header inside the answer row. Production has long mounted that header on the
enclosing Assistant-turn widget; the new passing mounted live control uses that
actual hierarchy. This correction changes body rendering but no header/turn
hierarchy, so those five unrelated assertions were not rewritten or claimed
green.

### Mutation, static, and privacy evidence

Every review mutant was applied alone and restored:

1. Limiting the new scan back to the active leaf failed all three recursive
   sole/duplicate/orphan ratchets (`3 failed, 3 deselected`).
2. Removing read-failure quarantine failed the exact blocking/re-read ratchet
   (`1 failed`).
3. Returning raw Markdown content instead of the shared state renderer failed
   all five mounted state variants (`5 failed, 2 deselected`).
4. Bypassing event-settled terminal recognition failed both real manual and
   queued full-controller ratchets (`2 failed, 4 deselected`).
5. Forcing ambiguous actions enabled failed the mounted disabled-action ratchet
   (`1 failed`).
6. Self-review found that scanning every sidecar also classified exact
   complete-continuation/complete-assistant history as invalid recovery. Its
   production-path ratchet first failed `1 failed, 6 deselected` with bounded
   `invalid_continuation`; removing the narrow terminal-history rule after the
   fix killed the same test with `1 failed, 6 deselected`, and the rule was
   restored.

`rg -n 'MUTATION PROBE'` over the changed production files returned no matches,
and the fresh 14/258/126/101/73/628 gates ran with restored production. The
post-self-review two-file focus was `14 passed, 1 warning in 2.87s`; the 258
gate includes that exact matrix.

```text
../../.venv/bin/python -m ruff check <changed production and review-test files>
All checks passed!

../../.venv/bin/python -m ruff format --check Tests/Chat/test_console_continuation_review_fixes.py Tests/UI/test_console_continuation_review_fixes.py
2 files already formatted

git diff --check
<empty>

git diff -U0 | rg -n '^\\+.*logger.*(content|prompt|evidence|checkpoint|provider_continuation_json|api[_-]?key)'
<empty>
```

Whole-file formatting remains qualified exactly as above: the inherited large
production modules are not formatter-clean at the approved parent, so no broad
mechanical rewrite is included. New warning/quarantine text is fixed and
content-free; no prompt, evidence, reasoning, continuation JSON, tool payload,
credential, or destination secret was added to logs or UI metadata.

### Review-fix files and self-review

Review-fix files:

- `Tests/Chat/test_console_continuation_review_fixes.py`
- `Tests/UI/test_console_continuation_review_fixes.py`
- `tldw_chatbook/Chat/console_dispatch_repository.py`
- `tldw_chatbook/Chat/console_chat_store.py`
- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Widgets/Console/console_transcript.py`
- `tldw_chatbook/UI/Console_Modules/provider_continuation_recovery.py`
- TASK-19900.3 notes, this report, and the shared progress ledger.

Self-review confirmed exact USER version/deleted quarantine and Task15 atomic
settlement remain intact; no new schema/table/transaction owner, USER/assistant,
fallback persistence, tool execution, or Task17 work was added. Recursive
continuation selection is unique/fail-closed, exact completed private history
is inert rather than actionable, unreadable hydration stays blocking, terminal
recognition requires matching durable state, and UI copy is bounded/literal.
TASK-19900.3 remains In Progress with all 22 criteria unchecked. No
Task16-scoped Important or Critical concern remains after this correction; only
the inherited Requests/format baselines and the five stale non-gating UI
assertions above remain qualified.
