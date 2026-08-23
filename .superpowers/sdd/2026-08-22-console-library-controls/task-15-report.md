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

## Fix round 1/5 — dispatch recovery ownership

### Authorities and scope

The round began from clean commit
`0c3e5bf76c14b95aa8783cb8d20e427cab6464af`. The Task 15 brief/report,
ADR-079, approved spec/plan sections, reviewer feedback, and testing, backlog,
and live-verification lessons were reread before production changes. No ADR is
required: this round closes correctness gaps inside ADR-079's existing
store/runtime/repository/UI ownership boundary. It adds no schema, transaction
owner, fallback persistence path, tool handoff, continuation execution, or Task
16 projection.

The approved `policy_revision=None` exception is deliberately narrow: only a
frozen `source=new_session, revision=None` authority may match a current
`source=durable, revision=1` authority, and every effective field must remain
equal. `temporary`, `missing`, and `unavailable` do not gain a durable-revision
wildcard.

### RED evidence

The six reported findings were reproduced through production controller, store,
repository, runtime, and mounted Textual paths before production edits:

```text
../../.venv/bin/python -m pytest Tests/Chat/test_console_dispatch_recovery_fix_round1.py Tests/UI/test_console_dispatch_recovery_fix_round1.py -q --tb=line --disable-warnings --show-capture=no
10 failed, 1 warning in 3.29s
```

The failures proved: a restored owner admitted a second submission; the
repository admitted a second active conversation checkpoint; mounted ChatScreen
did not expose the recovery region or the same-state Send gate; terminal
settlement rollback did not restore blocked runtime/queue ownership; Automatic
evidence was falsely unreconstructable; `revision=None` matched arbitrary later
durable revisions; and ordinary controller/store replacement and close erased
unresolved ephemeral ownership.

Diff self-review added four production ratchets before their fixes. After
correcting one initially wrong test label (`Retry response` is the model-owned
accepted-state copy), the exact combined RED was:

```text
../../.venv/bin/python -m pytest Tests/Chat/test_console_dispatch_recovery_fix_round1.py Tests/UI/test_console_dispatch_recovery_fix_round1.py -q --tb=line --disable-warnings --show-capture=no
6 failed, 9 passed, 1 warning in 6.68s
```

The three repository parameters showed that exact, changed attachment
`display_name`, and changed contribution replays were all accepted by an
incomplete repository comparison. The other failures proved a healthy durable
accepted owner was briefly recovery-visible before checkpoint transition, the
mounted queue shelf duplicated recovery controls, and a stale Button event
re-resolved the newly active session instead of its displayed owner.

### Implementation and contract decisions

- `ConsoleDispatchRecoveryState` now owns independent `runtime_active` and
  `recovery_needed` truth. Normal durable/ephemeral acceptance is runtime-active,
  hidden from recovery presentation, and still blocks another submission.
  Delivery-unknown or failed settlement restores the same exact owner as
  runtime-inactive, recovery-needed state.
- Controller admission checks the same neutral store gate used by mounted Send,
  before echo, preparation, acceptance, or provider resolution. The repository
  also rejects every pre-existing conversation checkpoint inside the caller's
  existing `BEGIN IMMEDIATE` boundary. Task 14's preparation fingerprint/cache
  remains the only exact production idempotence owner; the repository does not
  add a second partial fingerprint.
- Original-send success/failure/cancel and queued/manual terminal settlement
  errors use the dedicated `ConsoleDispatchSettlementError` barrier. It restores
  the exact assistant baseline, publishes BLOCKED recovery truth, force-hydrates
  the exact queued fence, and returns without legacy failure publication,
  continuation retirement, recovery cleanup, or queue advancement.
- Automatic retrieval evidence is reconstructable because Retry re-queries the
  durable USER plus frozen scope. Only explicit transient staged evidence and
  one-shot prefill remain unreconstructable.
- Ordinary close and state/controller replacement preserve unresolved
  store-owned ephemeral recovery. Only terminal settlement, Discard, or explicit
  `ConsoleRuntime.dispose()` app teardown clears it.
- ChatScreen always mounts one dedicated literal recovery region, including with
  zero queued entries. Queue UI retains count and a disabled `Resume` with
  `Paused for response recovery`, but never duplicates Retry/Discard controls.
  Every recovery Button carries its displayed session and assistant owner into
  the callback; no click re-resolves `active_session_id`.

### GREEN and adjacent verification

Fresh post-restoration Task 15 focus:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_dispatch_recovery.py Tests/Chat/test_console_dispatch_queue_recovery.py Tests/UI/test_console_dispatch_recovery.py Tests/Chat/test_console_dispatch_recovery_fix_round1.py Tests/UI/test_console_dispatch_recovery_fix_round1.py
68 passed, 1 warning in 14.26s
```

Task 14 acceptance/fix-round/first-send gate:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_durable_turn_acceptance.py Tests/Chat/test_console_durable_turn_fix_round1.py Tests/Chat/test_console_durable_turn_fix_round2.py Tests/Chat/test_console_durable_turn_fix_round3.py Tests/Chat/test_console_durable_turn_fix_round4.py Tests/Chat/test_console_first_send_atomicity.py
73 passed, 1 warning in 35.68s
```

Exact Task 13 affected gate:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Architecture/test_console_wave6_inventory.py Tests/Chat/test_console_automatic_library_preparation.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_turn_library_authority.py Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_turn_preparation.py Tests/Chat/test_library_preparation.py Tests/UI/test_console_auto_rag_on_send.py Tests/UI/test_console_harness_config_honesty.py Tests/UI/test_console_rag_settings_modal.py Tests/UI/test_console_retrieval_controller.py Tests/UI/test_console_controller_wiring.py Tests/test_config_console_defaults.py
628 passed, 1 warning in 37.26s
```

Queue and mounted UI gate:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_prompt_queue.py Tests/UI/test_console_prompt_queue.py Tests/UI/test_console_prompt_queue_modal.py Tests/UI/test_console_dispatch_recovery.py Tests/UI/test_console_dispatch_recovery_fix_round1.py
98 passed, 1 warning in 16.53s
```

Repository/transaction/model/hydration/state gate:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py Tests/Chat/test_console_transaction_contribution.py Tests/Chat/test_console_chat_models.py Tests/Chat/test_console_conversation_hydration.py Tests/CI/test_textual_runtime_contract.py
126 passed, 1 warning in 19.00s
```

Runtime ownership, excluding one independently stale app fixture:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/UI/test_console_runtime_ownership.py -k 'not app_fences_console_then_drains_buddy_before_profile_teardown'
10 passed, 1 deselected, 7 warnings in 20.98s
```

The excluded node fails unchanged before Console shutdown because its
`object.__new__(TldwCli)` fixture lacks `notes_sync_runtime_owner`; run alone it
was `1 failed, 1 warning in 3.32s`. Neither `app.py` nor that fixture changed.
No full repository suite, profile database, app launch, network provider, or
push was used.

### Mutation evidence

Every mutant was applied alone and restored immediately. The six report
findings killed these ratchets:

1. Removing controller admission made the restored-owner submit test enter
   persistence; removing the in-transaction repository guard made the second
   checkpoint test fail `DID NOT RAISE`.
2. Suppressing recovery presentation failed the mounted visibility test;
   projecting healthy live ownership as recovery failed its hidden-state
   assertion; disabling the same-state Send gate failed the mounted Button
   assertion.
3. Replacing `mark_dispatch_recovery_needed` with action release killed three
   success/failure/queued settlement cases because runtime/recovery truth stayed
   contradictory. An earlier hydration-only mutant survived and exposed that
   the test observed too late; the ratchet was strengthened to assert state and
   the queue fence at submit return before the killed owner-restore mutant.
4. Restoring the old Automatic evidence-unreconstructable condition disabled
   Retry and failed the real restart/re-query test.
5. Restoring `frozen_revision is None` as a wildcard failed the later-revision
   negative authority case while the exact first-save case remained the control.
6. Suppressing explicit app-runtime cleanup failed the teardown half of the
   ephemeral replacement/close/teardown lifecycle test.

The four review-added mutants produced exact focused output:

```text
# partial same-ID repository replay (exact/display-name/contribution)
3 failed, 1 warning in 0.75s
# infer healthy durable runtime truth only from in_flight
1 failed, 1 warning in 0.44s
# suppress the queue recovery-fence presentation
1 failed, 1 warning in 2.08s
# re-resolve active_session_id in the mounted callback
1 failed, 1 warning in 2.74s
```

The final mutant-marker scan returned no matches, and the two-file post-restore
ratchet passed `15 passed, 1 warning in 6.16s`.

### Static, UI, privacy, and self-review

```text
../../.venv/bin/python -m ruff check <12 changed Python production/test files>
All checks passed!

../../.venv/bin/python -m ruff format --check Tests/Chat/test_console_dispatch_recovery_fix_round1.py Tests/UI/test_console_dispatch_recovery_fix_round1.py tldw_chatbook/UI/Console_Modules/dispatch_recovery.py
3 files already formatted

node /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.agents/skills/impeccable/scripts/detect.mjs --json tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Console_Modules/dispatch_recovery.py tldw_chatbook/UI/Console_Modules/prompt_queue.py
[]

git diff --check
(no output, exit 0)
```

The qualified whole-file formatter command reports nine inherited large-file
drifts and three clean files; formatter diffs place the reported changes outside
this round's changed ranges. Source/privacy scans found no added logger/print,
credential/request-body text, schema DDL, provider-continuation, tool-batch, or
handoff seam. All dynamic recovery/queue strings are `markup=False`, and no
prompt, evidence body, provider secret, or credential was logged.

Self-review checked the complete production/test/governance diff, exact rollback
message count and DB state, source-local vs remote ownership, queue settlement
once-only behavior, mounted zero-count and queued layouts, stale-session action
identity, app teardown ordering, the repository lock boundary, and Task 14/16
ownership. The review found and fixed the four additional RED items above; no
unresolved Task 15 product concern remains. The inherited Requests warning,
stale runtime fixture, and whole-file formatter drift stay visible and were not
silenced or expanded.

### Files changed in fix round 1

Production:

- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Chat/console_chat_models.py`
- `tldw_chatbook/Chat/console_chat_store.py`
- `tldw_chatbook/Chat/console_dispatch_repository.py`
- `tldw_chatbook/Chat/console_runtime.py`
- `tldw_chatbook/UI/Console_Modules/dispatch_recovery.py`
- `tldw_chatbook/UI/Console_Modules/prompt_queue.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`

Tests:

- `Tests/Chat/test_console_dispatch_recovery_fix_round1.py`
- `Tests/UI/test_console_dispatch_recovery_fix_round1.py`
- `Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py`
- `Tests/Chat/test_console_transaction_contribution.py`

Governance/evidence:

- TASK-19900.3 Implementation Notes (status remains In Progress; 22 unchecked)
- `backlog/docs/lessons-testing-evidence.md`
- shared progress ledger
- this report

## Fix round 2/5 — postcommit interruption recovery

### Review findings and root cause

The round started from clean commit
`caa6b2b40fe8c780d1a865d5d8740d3e95deb035`. The review findings were
reproduced through real controller/store/repository paths before the first
round-2 production edit.

The first root cause was that generic exceptions after durable acceptance only
released an action claim until the checkpoint-transition effect had completed.
That left the already-committed owner runtime-active and recovery-hidden, and a
queued owner could return without its exact restart fence. The only live
continuation entry point was an internal method; the production Retry action
instead reconstructed from SQLite and bypassed unfinished queue acknowledgement,
hooks, history, and preparation publication.

The second root cause was a local boolean around Retry checkpoint transition.
Settlement failures after a normal return were recoverable, but a wrapper that
raised after the repository CAS and before the boolean assignment was
misclassified as pre-CAS. Review then exposed that the first RED matrix began
after owner publication: failures in `identity_publication` or
`durable_owner_publication` had no live assistant/recovery projection at all.
Finally, the first CAS relation compared only selected IDs/state/revision and
would authenticate an in-memory owner whose frozen authority, destination,
reconstructability, or message versions had changed.

### Corrected RED evidence

The official two-file RED command, against the round-1 baseline production,
was:

```text
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_dispatch_recovery_fix_round2.py Tests/UI/test_console_dispatch_recovery_fix_round2.py --tb=short
14 failed, 2 passed, 1 warning in 4.69s
```

Those failures covered seven post-owner effects, manual and queued recovery,
controller replacement, mounted Retry, and post-CAS success/failure settlement
faults. The cancellation and pre-CAS refusal controls passed. The mounted
fixture was corrected before this official run; no fixture/setup failure is
counted as RED evidence.

Review expanded the matrix to the two first postcommit effects and the exact
CAS fault boundary. Before the corresponding production changes:

```text
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_dispatch_recovery_fix_round2.py -k 'identity_publication or durable_owner_publication or retry_exception_after_checkpoint_cas' --tb=short --show-capture=no
3 failed, 15 deselected, 1 warning in 0.77s
```

Both early-effect cases had no recovery owner; the CAS-then-local-exception case
had a committed `dispatch_started` row but runtime-active store truth. A final
review ratchet changed frozen authority after CAS:

```text
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_dispatch_recovery_fix_round2.py -k 'retry_does_not_restore_a_checkpoint_mutated_after_cas' --tb=short --show-capture=no
1 failed, 18 deselected, 1 warning in 0.47s
```

The incomplete comparator invoked recovery restoration for that changed owner.

### Implementation and contract decisions

- Every exception after durable acceptance now restores the same committed
  owner as runtime-inactive/recovery-needed, sets BLOCKED truth, force-hydrates
  a queued fence, and retains the exact queue acceptance before returning.
- `publish_durable_recovery_owner` hydrates the committed USER/assistant and
  accepted checkpoint without completing any postcommit ledger effect. This is
  deliberately distinct from `publish_durable_turn_owners`: failures in either
  of the first two effects become actionable, while explicit Retry still resumes
  the failed effect itself and every later unfinished effect in order.
- The live continuation remains bounded to the existing preparation fingerprint
  and exact session/assistant/checkpoint/origin/queue owner. Production Retry
  invokes `resume_durable_postcommit`; a replacement controller that lacks the
  live continuation fails closed instead of bypassing the ledger or creating
  another message pair.
- Queue acknowledgement is not special-cased or skipped. A failed queued
  acknowledgement retains the claimed head and pauses later work; successful
  Retry completes it once, atomically settles the assistant/checkpoint, then
  drains later work once.
- Retry exception classification authenticates the complete legal CAS relation
  at catch time. The prior checkpoint must be accepted or dispatch-started; the
  new bounded attempt must differ; state becomes dispatch-started; checkpoint
  revision and assistant version increment exactly once; and dataclass equality
  keeps USER version, all IDs, frozen authority, credential-free destination,
  reconstructability, origin, and queue owner unchanged. There is no local
  boolean bypass.
- Pre-CAS refusal/cancellation only releases the action. Post-CAS provider
  success, failure, cancellation, settlement exception, or local wrapper
  exception restores runtime-inactive recovery truth and preserves the queue
  fence. No second USER/assistant, checkpoint, schema, transaction owner,
  persistence fallback, or Task-16 continuation/tool behavior was added.

ADR required: no.  ADR path:
`backlog/decisions/079-console-library-conversation-authority.md`.  Reason: this
round closes review gaps in ADR-079's already-approved exact-owner recovery and
does not change storage, sync, provider, or runtime boundaries.

### GREEN and adjacent verification

First GREEN for the original two-file matrix was `16 passed, 1 warning in
4.65s`. After the early-effect and exact-CAS review expansions, the same focused
matrix passed:

```text
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_dispatch_recovery_fix_round2.py Tests/UI/test_console_dispatch_recovery_fix_round2.py --tb=short --show-capture=no
20 passed, 1 warning in 5.56s
```

Fresh post-restoration Task-15 focus:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_dispatch_recovery.py Tests/Chat/test_console_dispatch_queue_recovery.py Tests/UI/test_console_dispatch_recovery.py Tests/Chat/test_console_dispatch_recovery_fix_round1.py Tests/UI/test_console_dispatch_recovery_fix_round1.py Tests/Chat/test_console_dispatch_recovery_fix_round2.py Tests/UI/test_console_dispatch_recovery_fix_round2.py
88 passed, 1 warning in 20.85s
```

Task-14 durable acceptance gate:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_durable_turn_acceptance.py Tests/Chat/test_console_durable_turn_fix_round1.py Tests/Chat/test_console_durable_turn_fix_round2.py Tests/Chat/test_console_durable_turn_fix_round3.py Tests/Chat/test_console_durable_turn_fix_round4.py Tests/Chat/test_console_first_send_atomicity.py
73 passed, 1 warning in 42.13s
```

Exact Task-13 affected gate:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Architecture/test_console_wave6_inventory.py Tests/Chat/test_console_automatic_library_preparation.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_turn_library_authority.py Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_turn_preparation.py Tests/Chat/test_library_preparation.py Tests/UI/test_console_auto_rag_on_send.py Tests/UI/test_console_harness_config_honesty.py Tests/UI/test_console_rag_settings_modal.py Tests/UI/test_console_retrieval_controller.py Tests/UI/test_console_controller_wiring.py Tests/test_config_console_defaults.py
628 passed, 1 warning in 45.91s
```

Queue and mounted-UI gate:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_prompt_queue.py Tests/UI/test_console_prompt_queue.py Tests/UI/test_console_prompt_queue_modal.py Tests/UI/test_console_dispatch_recovery.py Tests/UI/test_console_dispatch_recovery_fix_round1.py
98 passed, 1 warning in 21.03s
```

Repository/transaction/model/hydration/state gate:

```text
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py Tests/Chat/test_console_transaction_contribution.py Tests/Chat/test_console_chat_models.py Tests/Chat/test_console_conversation_hydration.py Tests/CI/test_textual_runtime_contract.py
126 passed, 1 warning in 23.50s
```

No full repository suite, profile database, app launch, provider network, or
push was used. Every pytest warning above is the inherited environment-level
requests/urllib3/charset compatibility warning.

### Mutation evidence

Each mutant was applied alone and restored immediately:

```text
# suppress early exact-owner recovery publication
2 failed, 17 deselected, 1 warning in 0.68s
# replace generic recovery restoration with action release
2 failed, 17 deselected, 1 warning in 0.66s
# bypass the live postcommit continuation from production Retry
2 failed, 18 deselected, 1 warning in 4.05s
# force every post-CAS exception to use the pre-CAS release path
5 failed, 14 deselected, 1 warning in 1.25s
# weaken complete checkpoint equality to state/revision/version only
1 failed, 18 deselected, 1 warning in 0.48s
```

The final post-restoration Task-15 command is the 88-pass gate above. Source
inspection found no mutant marker left in production.

### Static, privacy, and self-review

```text
../../.venv/bin/python -m ruff check tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_prompt_queue_coordinator.py Tests/Chat/test_console_dispatch_recovery_fix_round2.py Tests/UI/test_console_dispatch_recovery_fix_round2.py
All checks passed!

../../.venv/bin/python -m ruff format --check Tests/Chat/test_console_dispatch_recovery_fix_round2.py Tests/UI/test_console_dispatch_recovery_fix_round2.py
2 files already formatted

git diff --check
(no output, exit 0)
```

Whole-file format checking remains truthfully qualified: the inherited large
`console_chat_store.py` drift remains, and `console_chat_controller.py` reports
one pre-existing round-1 line outside this round's changed ranges;
`console_prompt_queue_coordinator.py` and both new tests are clean. Scoped diff,
source, and privacy inspection found no new log statement, credential/API-key,
prompt/evidence/body serialization, schema/migration, provider-continuation,
tool-batch, or Task-16 handoff seam.

Self-review checked all exception boundaries, exact equality fields, early
effect-ledger state, replacement-controller refusal, queue acknowledgement and
drain ordering, terminal rollback behavior, and message/checkpoint cardinality.
It found and fixed the two initially omitted early effects and the partial CAS
relation described above. No unresolved Task-15 product concern remains.

### Files changed in fix round 2

Production:

- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Chat/console_chat_store.py`
- `tldw_chatbook/Chat/console_prompt_queue_coordinator.py`

Tests:

- `Tests/Chat/test_console_dispatch_recovery_fix_round2.py`
- `Tests/UI/test_console_dispatch_recovery_fix_round2.py`

Governance/evidence:

- TASK-19900.3 Implementation Notes (status remains In Progress; 22 unchecked)
- shared progress ledger
- this report

## Fix round 4/5 — exception-safe close and volatile disposal cleanup

### Review findings and baseline

Round 4 resumed at clean `52ca85e0b8f7673c84342090020aa82fcc656ed3`.
ADR required: no. ADR path:
`backlog/decisions/079-console-library-conversation-authority.md`. Reason: this
round hardens ADR-079's existing app-runtime teardown boundary without changing
schema, transaction ownership, sync, provider execution, or Task-16 behavior.

The clean round-2/round-3 baseline was:

```text
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_dispatch_recovery_fix_round2.py Tests/UI/test_console_dispatch_recovery_fix_round2.py Tests/Chat/test_console_dispatch_recovery_fix_round3.py --tb=short --show-capture=no
30 passed, 1 warning in 7.71s
```

The complete new production-path matrix was authored before production edits.
It uses real temporary SQLite, a real accepted identity-publication
interruption, the exact staged EvidenceBundle lease, `close_session`,
`ConsoleRuntime.dispose`, and loader reconciliation:

```text
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_dispatch_recovery_fix_round4.py --tb=short --show-capture=no
2 failed, 1 passed, 1 warning in 0.85s
```

The intended failures were: a raising exact evidence release escaped
`close_session` before cache/preparation/session cleanup; and permanent runtime
disposal retained an actionable durable recovery projection. The exact-once,
replacement-preserving normal close control passed. No fixture/setup failure
was counted as RED.

Self-review strengthened the disposal ratchet with an independently populated
queued hydration fence and claimed-action message baseline. With only the
hydration clear removed, it failed at the exact stale-fence assertion:

```text
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_dispatch_recovery_fix_round4.py -k app_disposal --tb=short --show-capture=no
1 failed, 2 deselected, 1 warning in 0.47s
```

### Implementation and contract decisions

- `close_session` now retires a durable continuation through the existing
  exception-safe evidence retirement seam. A release callback fault scrubs the
  frozen lease references and cannot skip body/cache, preparation/outcome, or
  session cleanup. Normal release still targets the original frozen launch
  exactly once and preserves a replacement staged launch.
- Permanent `end_app_runtime` now clears all store-owned volatile dispatch
  recovery projections, claimed-action message baselines, and queued hydration
  fences, durable and ephemeral alike. SQLite checkpoints are unchanged, so a
  newly constructed store/loader reconstructs the exact durable recovery on
  restart. Ordinary navigation remains outside this permanent teardown seam.
- The changes add no checkpoint mutation, message mutation, provider call,
  schema, persistence fallback, prompt/evidence logging, or Task-16
  continuation/tool behavior.

### GREEN and adjacent verification

First GREEN was `3 passed, 1 warning in 0.73s`; the strengthened final round-4
file was `3 passed, 1 warning in 0.85s`. Fresh post-restoration gates were:

```text
# Task 15 focus (base, queue, UI, fix rounds 1-4)
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_dispatch_recovery.py Tests/Chat/test_console_dispatch_queue_recovery.py Tests/UI/test_console_dispatch_recovery.py Tests/Chat/test_console_dispatch_recovery_fix_round1.py Tests/UI/test_console_dispatch_recovery_fix_round1.py Tests/Chat/test_console_dispatch_recovery_fix_round2.py Tests/UI/test_console_dispatch_recovery_fix_round2.py Tests/Chat/test_console_dispatch_recovery_fix_round3.py Tests/Chat/test_console_dispatch_recovery_fix_round4.py
101 passed, 1 warning in 25.29s

# Task 14 durable acceptance gate
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_durable_turn_acceptance.py Tests/Chat/test_console_durable_turn_fix_round1.py Tests/Chat/test_console_durable_turn_fix_round2.py Tests/Chat/test_console_durable_turn_fix_round3.py Tests/Chat/test_console_durable_turn_fix_round4.py Tests/Chat/test_console_first_send_atomicity.py
73 passed, 1 warning in 37.02s

# Exact Task 13 affected gate
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Architecture/test_console_wave6_inventory.py Tests/Chat/test_console_automatic_library_preparation.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_turn_library_authority.py Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_turn_preparation.py Tests/Chat/test_library_preparation.py Tests/UI/test_console_auto_rag_on_send.py Tests/UI/test_console_harness_config_honesty.py Tests/UI/test_console_rag_settings_modal.py Tests/UI/test_console_retrieval_controller.py Tests/UI/test_console_controller_wiring.py Tests/test_config_console_defaults.py
628 passed, 1 warning in 41.09s

# Queue and mounted UI
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_prompt_queue.py Tests/UI/test_console_prompt_queue.py Tests/UI/test_console_prompt_queue_modal.py Tests/UI/test_console_dispatch_recovery.py Tests/UI/test_console_dispatch_recovery_fix_round1.py
98 passed, 1 warning in 20.18s

# Repository/transaction/model/hydration/state
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py Tests/Chat/test_console_transaction_contribution.py Tests/Chat/test_console_chat_models.py Tests/Chat/test_console_conversation_hydration.py Tests/CI/test_textual_runtime_contract.py
126 passed, 1 warning in 23.16s

# App-runtime lifetime companion
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_runtime_lifetime.py
14 passed, 1 warning in 2.66s
```

Every successful pytest command reported only the inherited environment-level
Requests/urllib3/charset compatibility warning. No full repository suite,
profile database, app launch, provider network, or push was used.

### Mutation evidence

Each mutant was applied alone and restored immediately:

```text
# replace close retirement with the raw raising evidence-release call
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_dispatch_recovery_fix_round4.py -k close_session --tb=short --show-capture=no
1 failed, 1 passed, 1 deselected, 1 warning in 0.68s

# omit the durable/ephemeral recovery projection clear
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_dispatch_recovery_fix_round4.py -k app_disposal --tb=short --show-capture=no
1 failed, 2 deselected, 1 warning in 0.44s

# omit the queued recovery hydration-fence clear
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_dispatch_recovery_fix_round4.py -k app_disposal --tb=short --show-capture=no
1 failed, 2 deselected, 1 warning in 0.47s
```

After restoration, the round-4 file passed all 3 tests and source inspection
confirmed the hardened release call plus all three volatile clears are present;
no mutant marker remains.

### Static, privacy, and self-review

```text
../../.venv/bin/python -m ruff check tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_console_dispatch_recovery_fix_round4.py
All checks passed!

../../.venv/bin/python -m ruff format --check Tests/Chat/test_console_dispatch_recovery_fix_round4.py
1 file already formatted

git diff --check
(no output, exit 0)

git diff -U0 -- tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_chat_store.py | rg '^\\+.*(logger\\.|print\\(|api[_-]?key|authorization|bearer|credential|CREATE (TABLE|INDEX|TRIGGER)|ALTER TABLE|tool[_-]?batch|continuation_active|prompt|request[_-]?body|capture_result)'
(no output, exit 1)
```

Whole-file Ruff formatting remains truthfully qualified: clean HEAD already
fails formatting for both large production files; this round's new test is
formatted and the two tiny production ranges were inspected as formatter-clean.
Self-review checked callback exception ordering, exact frozen/replacement
evidence identity, preparation/outcome/cache/session cardinality, permanent
versus navigation lifetime, recovery/action/baseline/hydration clearing,
unchanged SQLite checkpoint bytes, loader rehydration, zero provider calls,
privacy, and Task-14/16 boundaries. No unresolved round-4 product concern
remains.

### Files changed in fix round 4

Production:

- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Chat/console_chat_store.py`

Tests:

- `Tests/Chat/test_console_dispatch_recovery_fix_round4.py`

Governance/evidence:

- TASK-19900.3 Implementation Notes (status remains In Progress; 22 unchecked)
- shared progress ledger
- this report

## Fix round 3/5 — bounded teardown and prerequisite-safe Discard

### Review findings and root cause

This round started at clean commit
`df47189ddd7fc24b12fc4c0a3128e23c4bc03523`. The Task-15 brief/report,
ADR-079/spec/plan, TASK-19900.3, reviewer findings, and testing/live/backlog
lessons were reread before test authoring. The exact round-2 baseline was:

```text
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_dispatch_recovery_fix_round2.py Tests/UI/test_console_dispatch_recovery_fix_round2.py --tb=short --show-capture=no
20 passed, 1 warning in 5.99s
```

The first root cause was the no-task early return in `shutdown()`: permanent app
disposal did not retire a live durable postcommit continuation or the store's
two content-bearing durable caches. The volatile preparation could also remain
COMMITTING. The second root cause was that continuation retirement never
released a prepared explicit-evidence lease. The release lookup also depended
on the current mounted retrieval hook, but real `ConsoleRuntime.dispose()`
detaches that hook before controller shutdown, so teardown could not release
the original owner. The third root cause was that Discard atomically settled
the checkpoint immediately, bypassing unfinished identity/owner/input/queue/
hook/history/preparation effects and then dropping the only live continuation.
An early failure could therefore orphan the committed conversation and leave a
COMMITTING preparation that wedged the next send.

### RED evidence

The complete initial round-3 matrix was authored before the first production
edit and failed only at the intended production assertions:

```text
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_dispatch_recovery_fix_round3.py --tb=short --show-capture=no
8 failed, 1 warning in 1.67s
```

Those eight failures cover no-task app-disposal retention; four real
identity/owner/staged-clear/preparation-publication Discard boundaries;
prerequisite failure retention and later action retry; alien-controller
fail-closed behavior; exact draft/session identity cleanup; COMMITTING owner
retirement; next-send admission; and exact evidence identity/replacement/
idempotence.

The runtime-lifecycle companion then exposed one further real boundary: after
`dispose()` detached the retrieval hook, the newly required app-teardown
retirement had no way to release its exact frozen evidence owner. That
additional ratchet was RED before the release-capability change:

```text
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_dispatch_recovery_fix_round3.py --tb=short --show-capture=no
1 failed, 8 passed, 1 warning in 1.92s
```

The failure was the exact original frozen launch remaining unreleased; it was
not a fixture or setup failure.

Self-review then injected a raising exact-release callback at the real app
disposal boundary. Before the final cleanup guard, the callback exception
prevented durable-cache retirement:

```text
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_dispatch_recovery_fix_round3.py -k release_raises --tb=short --show-capture=no
1 failed, 9 deselected, 1 warning in 0.47s
```

The final retirement helper treats that already-terminal/app-disposal UI fault
as non-retryable, scrubs the lease references, and continues body/cache cleanup.

### Implementation and contract decisions

- Permanent controller shutdown now retires every remaining live durable
  postcommit continuation after task teardown, including the prior no-task
  early-return path. It removes the exact volatile preparation/outcome/sidecar,
  releases evidence, and retires content-bearing store caches into the existing
  bounded body-free tombstone. The durable accepted checkpoint remains for
  restart recovery and no provider is invoked.
- The live-only evidence lease binds the original retrieval owner's exact
  release capability at admission. Release therefore survives navigation and
  app-view detachment, targets only the frozen launch, preserves a newer launch,
  and is idempotent. After successful release the lease clears its launch,
  capture result, and callback references immediately.
- A live pretransition Discard authenticates the same fingerprint/session/
  assistant/checkpoint/origin/queue continuation, resumes every unfinished
  required postcommit effect through `preparation_publication`, and stops before
  checkpoint CAS/provider entry. It then reclaims the exact recovery owner and
  uses the existing atomic assistant/checkpoint settlement. No USER, assistant,
  checkpoint, provider call, persistence fallback, or Task-16 continuation is
  created.
- A prerequisite failure restores the same blocked actionable recovery and
  queue fence; a later Discard retries unfinished idempotent effects. A
  replacement controller lacking the live continuation fails closed. Successful
  prerequisite publication moves the volatile preparation out of COMMITTING so
  retirement removes it and the next send is not wedged.
- The shutdown lifecycle documentation now states the production ownership
  truth: `ConsoleRuntime.leave_console` is ordinary navigation and reuses the
  app-owned controller; `shutdown()` is permanent app disposal.

ADR required: no. ADR path:
`backlog/decisions/079-console-library-conversation-authority.md`. Reason: this
round closes exact-owner teardown and settlement gaps within ADR-079's existing
app-runtime recovery boundary; it adds no schema, sync, provider, or Task-16
contract.

### GREEN and adjacent verification

The initial matrix first reached `8 passed, 1 warning in 1.81s`; after the
exact teardown-release and self-review fault expansions, final round-3 focus
was `10 passed, 1 warning in 2.10s`. Fresh post-restoration gates were:

```text
# Task 15 focus (base, queue, UI, fix rounds 1-3)
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_dispatch_recovery.py Tests/Chat/test_console_dispatch_queue_recovery.py Tests/UI/test_console_dispatch_recovery.py Tests/Chat/test_console_dispatch_recovery_fix_round1.py Tests/UI/test_console_dispatch_recovery_fix_round1.py Tests/Chat/test_console_dispatch_recovery_fix_round2.py Tests/UI/test_console_dispatch_recovery_fix_round2.py Tests/Chat/test_console_dispatch_recovery_fix_round3.py
98 passed, 1 warning in 21.84s

# Task 14 durable acceptance gate
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_durable_turn_acceptance.py Tests/Chat/test_console_durable_turn_fix_round1.py Tests/Chat/test_console_durable_turn_fix_round2.py Tests/Chat/test_console_durable_turn_fix_round3.py Tests/Chat/test_console_durable_turn_fix_round4.py Tests/Chat/test_console_first_send_atomicity.py
73 passed, 1 warning in 36.10s

# Exact Task 13 affected gate
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Architecture/test_console_wave6_inventory.py Tests/Chat/test_console_automatic_library_preparation.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_turn_library_authority.py Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_turn_preparation.py Tests/Chat/test_library_preparation.py Tests/UI/test_console_auto_rag_on_send.py Tests/UI/test_console_harness_config_honesty.py Tests/UI/test_console_rag_settings_modal.py Tests/UI/test_console_retrieval_controller.py Tests/UI/test_console_controller_wiring.py Tests/test_config_console_defaults.py
628 passed, 1 warning in 39.51s

# Queue and mounted UI
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_prompt_queue.py Tests/UI/test_console_prompt_queue.py Tests/UI/test_console_prompt_queue_modal.py Tests/UI/test_console_dispatch_recovery.py Tests/UI/test_console_dispatch_recovery_fix_round1.py
98 passed, 1 warning in 17.19s

# Repository/transaction/model/hydration/state
../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py Tests/Chat/test_console_transaction_contribution.py Tests/Chat/test_console_chat_models.py Tests/Chat/test_console_conversation_hydration.py Tests/CI/test_textual_runtime_contract.py
126 passed, 1 warning in 18.94s
```

The inherited warning in each successful command is the environment-level
Requests/urllib3/charset compatibility warning. No full repository suite,
profile database, app launch, network provider, or push was used.

The additional runtime companion command produced `24 passed, 1 failed, 7
warnings in 25.87s`; the one node reproduced alone as a two-second timeout
because its app shutdown task raised `AttributeError` for the fixture's missing
`notes_sync_runtime_owner` before entering Console disposal. The 24 actual
runtime/controller lifecycle nodes passed. This independently stale fixture is
kept visible and was not changed, deselected, or counted as product GREEN.

### Mutation evidence

Each mutant was applied alone and restored immediately:

```text
# remove no-task shutdown continuation retirement
1 failed, 7 deselected, 1 warning in 0.46s

# remove the exact frozen evidence release capability
2 failed, 7 deselected, 1 warning in 0.66s

# bypass live Discard prerequisite authentication/resumption
6 failed, 3 deselected, 1 warning in 1.32s
```

After restoration, the round-3 file passed 10 tests and source inspection
confirmed `release = lease.release` was restored with no `False and` or mutant
marker in production. The two `lease.release = None` assignments are the
intentional post-release/fault-path privacy scrubs, not the forced-release
mutant.

### Static, privacy, and self-review

```text
../../.venv/bin/python -m ruff check tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_dispatch_recovery_fix_round3.py
All checks passed!

../../.venv/bin/python -m ruff format --check Tests/Chat/test_console_dispatch_recovery_fix_round3.py
1 file already formatted

git diff --check
(no output, exit 0)

git diff -U0 -- tldw_chatbook/Chat/console_chat_controller.py | rg '^\\+.*(logger\\.|print\\(|api[_-]?key|authorization|bearer|credential|CREATE (TABLE|INDEX|TRIGGER)|ALTER TABLE|tool[_-]?batch|continuation_active)'
(no output, exit 1)
```

Whole-file controller format remains truthfully qualified: Ruff reports only
the inherited round-1 wrapping drift at the pre-existing submission-gate copy;
the round-3 production ranges and new test file are formatter-clean. Self-review
checked the durable fingerprint/owner relation, effect ordering, queue
acknowledgement before settlement, checkpoint/provider exclusion, settlement
failure retention, preparation removal, exact evidence callback lifetime,
callback-fault cleanup, body-free teardown, database/message cardinality, and
Task-14/16 boundaries.
No new logger, request/prompt/evidence body metadata, credential, DDL, provider
fallback, or continuation/tool handoff was added. No unresolved round-3 product
concern remains.

### Files changed in fix round 3

Production:

- `tldw_chatbook/Chat/console_chat_controller.py`

Tests:

- `Tests/Chat/test_console_dispatch_recovery_fix_round3.py`

Governance/evidence:

- TASK-19900.3 Implementation Notes (status remains In Progress; 22 unchecked)
- shared progress ledger
- this report
