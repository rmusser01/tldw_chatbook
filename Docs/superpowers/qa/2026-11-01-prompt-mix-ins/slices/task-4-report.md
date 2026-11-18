# Task 4 — runtime selection and admission capture

Status: complete implementation, pending controller review. Backlog TASK-38539 remains **In Progress**; AC 1–4 checked only after evidence. BASE `d203850171f4fa2150ad2540ae9f56486ba31614`. Commit: `32df824beb` (`feat: freeze mix-ins during Console admission`).

## Implementation and owner coverage

- Added `ConsolePromptMixinService.preferences`, `replace_preferences`, `capture`, and `retry_snapshot` on the existing contribution module. Live handles retain weak exact session/incarnation/binding witnesses; store CAS rejects stale revisions and replacement sessions. Pure placement reconciliation moves observed changes last in the destination group and preserves disabled order/flags.
- `ConsoleSessionInputSnapshot` freezes references/revision at synchronous received Send/Queue intake, before the runtime driver’s first await. Received claims, native configuration session identity, and receipt source guards refuse stale preferences/source/session before promotion. The real original Workspace SQL worker is held in a regression while preferences advance; it refuses and preserves the draft. Another real received path holds A’s source read, rejects a duplicate A Send, switches to B and changes B, then admits only A’s captured reference.
- `PromptScopeService.mixin_source_identity(mode=...)` is the shared synchronous identity seam. Local identity is `local:` plus SHA-256 of the already-selected lexical Prompt DB path, with exact local-service/DB object fences across awaits. It survives file reopen, changes on retarget, and exposes no path/credentials. Weak process-memory nonces identify memory databases; current server capture remains unsupported. Real local source reads use the existing finite preparation-read owner under a five-second capture deadline and retire physically before cancellation returns.
- Added representation-hidden, default-empty `ConsoleTurnConfigurationSnapshot.prompt_mixins`, included in detached execution copies and existing request/queue custody. No configuration JSON recovery codec exists in these owners; no serializer/sidecar/persistent queue was invented. The existing preparation and runtime dataclass/request copies preserve this field. Queue text edits preserve the admitted snapshot while other edit-time configuration behavior remains. Real coordinator edit → toggle/delete → drain retains the original body without another source read. New automatic origins and hook/native follow-ups explicitly receive empty values; continuations within an accepted envelope retain it.
- The legacy direct-submit path freezes mix-ins plus a private input witness before its first hook/config await, then inserts only this added domain at the **original** configuration-resolution point. Already-complete supplied configuration/custody is authoritative, including explicit/default-empty snapshots; custom complete callers use the preparation seam or supply their own exact snapshot. No sentinel/recapture inference was added.
- Selection saves reuse the app’s existing `ConsoleSettingsDurabilityOwner` and store’s coalescing settings drain as `PROMPT_MIXINS`; rapid edits share one app-owned save task. Exact binding/incarnation fences protect pending saves. Failures keep live choices and use the existing component ledger, exact failure/revision retry, and a hidden-when-healthy Model recovery row. No new timer, manager, or normal selection control. Settings Apply still owns its original two components, so those fixture/expectation enumerations were narrowed explicitly. Hydration reads preferences in the existing off-loop batch; private session-state round trips preserve unavailable formats. Temporary promotion seeds supported preferences using the existing restricted INSERT contribution in its original transaction.
- Retry is independent of the library. Temporary recorded snapshots use the session-private memory owner. Durable lookup checks raw strict markers and the exact assistant anchor before repository replay; recorded-empty remains distinct, truly legacy maps empty, and missing/partial/foreign new-format records refuse. Native marker-preservation triggers remain intact.

ADR required: **no**. Existing [ADR-286](../../../backlog/decisions/286-console-prompt-mix-ins.md) governs these boundaries; no new schema, dependency, cross-source transaction or source endpoint. Controller rulings explicitly approved the shared lexical source identity, failure-only existing-pattern rail row, queue-edit retention and complete-input compatibility. The task AC/plan was extended before the omitted rail path; existing hydration/persistence owners were extended only where needed. Composition/provider dispatch and durable attempt contribution integration remain Task 5; normal Prompts/Workbench controls remain later slices.

## Tests and evidence

All commands ran from the managed worktree with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`, `-p no:cacheprovider`, and private `--basetemp`. Counts below overlap; no aggregate unique total is claimed.

### TDD RED

`python -m pytest Tests/Chat/test_console_prompt_mixin_admission.py -q -p no:cacheprovider --basetemp=/private/tmp/mixin-task4-red`

**1 failed, 8 errors**: `reconcile_mixin_placements` and `ConsolePromptMixinService` did not exist. Expected missing runtime behavior before implementation. Initial GREEN was **9 passed**.

Additional native retry RED: `... test_console_prompt_mixin_admission.py::test_native_retry_classification_survives_unparsed_metadata ... --basetemp=/private/tmp/mixin-task4-retry-red` → **4 failed, 2 passed**; an unparsed new-format marker incorrectly became legacy empty. Final native six-state control passed **6/6** (`--basetemp=/private/tmp/mixin-task4-retry-native-final`).

### Final focused proof

```text
python -m pytest Tests/Chat/test_console_prompt_mixin_admission.py \
 Tests/Chat/test_console_turn_execution_context.py::test_attachment_gate_and_payload_use_captured_capabilities \
 Tests/Chat/test_console_received_intent_transitions.py::test_mixin_change_during_original_configuration_worker_refuses_draft \
 Tests/Chat/test_console_received_intent_transitions.py::test_received_mixin_capture_holds_chat_a_while_b_choices_change \
 Tests/UI/test_polling_recovery_status_updates.py::test_mixin_save_recovery_is_hidden_when_healthy_and_exact_when_failed \
 -q -p no:cacheprovider --basetemp=/private/tmp/mixin-task4-final-proof --tb=short --show-capture=no
```

**34 passed in 9.76s; no warning summary.** [Full retained output](task-4-evidence/final-proof.log). Covers order, current-version/placement changes, source authority/replacement/retarget, disabled missing/enabled missing, stale revisions/incarnations, unknown preferences, native save/reopen/promotion, app save coalescing/admission, save failure/stale retry, finite native source cancellation, memory/native exact retry classification, queue copies/edit/coordinator drain, complete supplied empty compatibility, real received races and the recovery row.

After adding the explicit duplicate A receipt assertion to the existing held real A/B source test, that exact node passed **1/1 in 1.72s**, `--basetemp=/private/tmp/mixin-task4-two-received-race`.

### Connected owners

- `python -m pytest Tests/Chat/test_console_received_admission.py Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_prompt_queue.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_configuration_preparation.py -q -p no:cacheprovider --basetemp=/private/tmp/mixin-task4-connected1` → **196 passed in 45.14s** before later targeted refinements.
- `python -m pytest Tests/Chat/test_console_prompt_mixin_admission.py Tests/Chat/test_console_received_intent_transitions.py Tests/Chat/test_console_received_admission.py Tests/Chat/test_console_prompt_queue.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_configuration_preparation.py Tests/Chat/test_console_configuration_worker_lifetime.py -q -p no:cacheprovider --basetemp=/private/tmp/mixin-task4-final-runtime` → **237 passed, 1 failed**. The failure was the introduced attachment regression described below, **not** inherited baseline noise.
- After correcting that regression: `python -m pytest Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_prompt_mixin_admission.py Tests/Chat/test_console_received_intent_transitions.py Tests/Chat/test_console_received_intent_custody.py -q -p no:cacheprovider --basetemp=/private/tmp/mixin-task4-final-context-custody` → **101 passed in 19.72s**. Subsequent replay-only logic has the final focused proof above.
- `python -m pytest Tests/Chat/test_console_settings_apply_store.py Tests/Chat/test_console_first_send_atomicity.py Tests/Chat/test_console_prompt_mixin_admission.py Tests/UI/test_polling_recovery_status_updates.py -q -p no:cacheprovider --basetemp=/private/tmp/mixin-task4-save-green` → **79 passed in 38.70s**.
- `python -m pytest Tests/Chat/test_console_prompt_mixin_admission.py Tests/Chat/test_console_conversation_hydration.py -q -p no:cacheprovider --basetemp=/private/tmp/mixin-task4-hydration-green` → **78 passed in 35.83s**.
- `python -m pytest Tests/Prompt_Management/test_prompt_scope_service.py Tests/Chat/test_console_settings_apply_store.py Tests/Chat/test_console_conversation_hydration.py Tests/UI/test_polling_recovery_status_updates.py Tests/UI/test_design_token_governance.py -q -p no:cacheprovider --basetemp=/private/tmp/mixin-task4-final-source-save-ui` → **285 passed in 60.49s**, no warning summary.
- `python -m pytest Tests/Chat/test_console_prompt_mixin_admission.py Tests/DB/test_console_prompt_mixins.py Tests/Chat/test_console_prompt_mixin_values.py Tests/Chat/test_console_transaction_contribution.py -q -p no:cacheprovider --basetemp=/private/tmp/mixin-task4-replay-green` → **119 passed, 3 failed, 1 aggregate FD-growth warning**. The three failures were the new replay fixture incorrectly mutating a detached store message projection rather than hydrating its real persisted anchor; that fixture was corrected and the six native replay nodes and final 34-node proof passed. The marker-rewrite control also initially expected corruption even though Task 3’s native trigger correctly restores the original pair; it now proves preserved replay, and a genuinely partial initial marker proves refusal. No trigger/admission reset or weaker native owner was used.

## Static and self-review

New/extended mix-in owner files and new admission tests pass `ruff check --no-cache` and `ruff format --no-cache --check`. Changed existing methods were formatted selectively. Exact-BASE Ruff code/message multiplicity comparison across every touched Python file: **667 baseline / 666 current, zero introduced diagnostics**; changed-line check also zero. [Differential evidence](task-4-evidence/ruff-differential.json). `git diff --check` passes.

Self-review traced the supplied/default-empty contract, exact session/source/CAS witnesses, same-ID recreation/rebinding, queue edit/drain and machine-origin construction, private body-bearing copies vs render projections, missing-record discriminator, settings-owner shutdown/coalescing and native retirement. Existing configuration runtime/store/admission fences remain intact. The initially early full configuration assignment in direct submit changed the pre-existing `custodied_inputs` branch and dropped pending attachments; it was removed in favor of the private snapshot/witness bridge until the original config owner. Its focused regression and whole affected context module pass.

## Concerns and excluded checks

The aggregate storage run emitted FD growth **268 (14→282, threshold 200)**. Prior Task 3 evidence also records aggregate FD growth (330 in its 336-test run); no new attribution was established and no FD cleanup/threshold/admission change was made. Final focused proof and source/settings/hydration/UI run have no warning summary. Existing large store/controller files remain large; this slice extends their established seams without restructuring them.

No full suite, live external provider/cache claims, normal Workbench UI, composition/dispatch expansion, new branch, push or merge. Those excluded feature slices are not claimed complete. No git stash/reset/clean or recovery-admission weakening.

## Files changed

- `Docs/Architecture/console.md`
- `Tests/Chat/test_console_first_send_atomicity.py`
- `Tests/Chat/test_console_received_intent_transitions.py`
- `Tests/Chat/test_console_settings_apply_store.py`
- `Tests/UI/test_polling_recovery_status_updates.py`
- `backlog/tasks/task-38539 - Capture-chat-mix-in-selections-during-manual-and-queue-admission.md`
- `tldw_chatbook/Chat/chat_persistence_service.py`
- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/Chat/console_chat_store.py`
- `tldw_chatbook/Chat/console_configuration_capture.py`
- `tldw_chatbook/Chat/console_configuration_preparation.py`
- `tldw_chatbook/Chat/console_conversation_hydration.py`
- `tldw_chatbook/Chat/console_prompt_mixin_service.py`
- `tldw_chatbook/Chat/console_prompt_mixins.py`
- `tldw_chatbook/Chat/console_prompt_queue.py`
- `tldw_chatbook/Chat/console_prompt_queue_coordinator.py`
- `tldw_chatbook/Chat/console_received_dispatch.py`
- `tldw_chatbook/Chat/console_received_intent.py`
- `tldw_chatbook/Chat/console_received_turn.py`
- `tldw_chatbook/Chat/console_turn_context.py`
- `tldw_chatbook/Prompt_Management/prompt_scope_service.py`
- `tldw_chatbook/UI/Console_Modules/left_rail.py`
- `tldw_chatbook/UI/Console_Modules/session.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `Tests/Chat/test_console_prompt_mixin_admission.py`

Commit completed locally; tracked worktree status is clean. Git emitted its existing disabled-auto-GC/unreachable-loose-object warnings; no shared repository housekeeping was attempted. Report/evidence remain in the existing ignored SDD artifact directory for controller review.


## Review fix round 1 — I1 (BASE `32df824beb`)

**Fixed:** direct uncustodied Queue/manual Submit now retain the service's body-free exact source/session/preference currentness closure alongside their private snapshot/witness. The same predicate serves source capture; direct paths check after the hook await and through preparation/provider waits immediately before admission. The owning loop rechecks after `drain_agent_handoff` before issuing the native save. Private `_PreparedSendContinuation.prompt_mixin_capture` is representation-hidden and keeps the same witness through pause/resume. Accepted/postcommit and supplied complete envelopes remain authoritative, and no callback enters frozen configuration/serialization or reloads bodies. Stale optimistic echoes become failed; COMMITTING cleanup uses the existing pause→abandon path after native ownership releases. Drafts/attachments remain staged and no preparation wedges.

Files: `Chat/console_prompt_mixin_service.py`, `Chat/console_chat_controller.py`, `Chat/console_native_commit.py`, the admission tests, Console architecture and Backlog notes. No ADR/schema/dependency/UI scope change.

All pytest commands use the existing venv Python, `-q -p no:cacheprovider`, private `--basetemp`, `--tb=short --show-capture=no`.

- **RED:** `python -m pytest Tests/Chat/test_console_prompt_mixin_admission.py::test_direct_admission_retains_source_across_hook_await ... --basetemp=/private/tmp/mixin-task4-fix1-red` → **8 failed in 2.00s**. Real PromptScope/local backend/DB replacement at the same path, and real lexical path retarget, all wrongly admitted Queue or manual work during held hooks. [Output](task-4-evidence/fix1-red.log).
- **GREEN:** the same eight nodes → **8 passed in 2.12s**, `--basetemp=/private/tmp/mixin-task4-fix1-green`.
- **Final focused proof:** `python -m pytest Tests/Chat/test_console_prompt_mixin_admission.py::test_direct_admission_retains_source_across_hook_await Tests/Chat/test_console_prompt_mixin_admission.py::test_direct_source_fence_survives_later_durable_awaits Tests/Chat/test_console_prompt_mixin_admission.py::test_pending_direct_continuation_keeps_source_fence Tests/Chat/test_console_prompt_mixin_admission.py::test_complete_supplied_empty_configuration_does_not_recapture Tests/Chat/test_console_prompt_mixin_admission.py::test_coordinator_drain_keeps_captured_mixins_after_delete Tests/Chat/test_console_turn_execution_context.py::test_attachment_gate_and_payload_use_captured_capabilities ... --basetemp=/private/tmp/mixin-task4-fix1-focused-green` → **15 passed in 5.04s**, no warning summary. Covers real source hook races, provider and registered native-owner drain barriers, draft/attachment/echo/preparation/native cleanup, pause/resume, accepted queue retention and original attachment timing. [Output](task-4-evidence/fix1-focused-green.log).
- **Connected owners:** `python -m pytest Tests/Chat/test_console_prompt_mixin_admission.py Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_native_commit.py Tests/Chat/test_console_native_commit_integration.py Tests/Chat/test_console_compaction_ask_hold.py::test_compact_and_send_summarizes_then_sends_the_held_message_once Tests/Chat/test_console_compaction_ask_hold.py::test_send_without_compacting_sends_with_no_summary_call ... --basetemp=/private/tmp/mixin-task4-fix1-final` → **141 passed, 1 failed in 54.28s**, no warning summary. The sole failure was a new assertion expecting `MixinSourceChangedError`; the existing `_continue_prepared_submission` deliberately converts it into a refused `ConsoleSubmitResult`. Corrected that assertion without changing the existing error contract; both continuation nodes passed **2/2** afterward. [Original covering output](task-4-evidence/fix1-final.log).
- **Latest cleanup refinement:** `python -m pytest Tests/Chat/test_console_prompt_mixin_admission.py::test_pending_direct_continuation_keeps_source_fence Tests/Chat/test_console_prompt_mixin_admission.py::test_direct_source_fence_survives_later_durable_awaits ... --basetemp=/private/tmp/mixin-task4-fix1-final-cleanup` → **4 passed**, no warning summary. This follows the final stale resumed-echo cleanup adjustment. [Output](task-4-evidence/fix1-cleanup-green.log). Counts overlap; no unique aggregate claimed.

**Static:** fix-BASE Ruff code/message multiplicities **182 baseline / 182 current; zero introduced**. [Evidence](task-4-evidence/fix1-ruff.json). `ruff check --no-cache` and `ruff format --no-cache --check` pass on the service and admission test file; `git diff --check` passes. Self-review kept guards owner-loop-affine at native handoff, callbacks private/repr-hidden, complete inputs authoritative, and acceptance cleanup within existing lifecycle owners. The additional held native-drain regression exposed a COMMITTING cancellation refusal; using the legal pause→abandon helper closed it. No new blocker or new resource warning. Earlier aggregate FD concern remains qualified as previously recorded. No broad 285-node repeat/full suite or unrelated changes.

Review fix commit: `a0c2c4bc32` (`fix: fence mix-in sources through direct admission`). Tracked worktree clean; Task remains In Progress. Existing Git auto-GC/unreachable-object warnings repeated, no housekeeping performed.
