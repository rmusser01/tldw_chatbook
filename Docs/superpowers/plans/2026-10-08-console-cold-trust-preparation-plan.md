# Console cold trust preparation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The integration owner controls implementation assignments and all native runs; this planning lane remains source-only.

**Goal:** Let a proven stock Console Send receive its draft immediately when only skill-trust initialization is cold, then perform that initialization under its existing runtime custody before normal checked configuration capture.

**Architecture:** Reuse the existing service-source proof and finite `ensure_local_skill_trust_service` initializer. Factor the common service proof out of the current skill-discovery adapter, keep that adapter's App-worker contract, and add a received-turn adapter using the existing custody task and preparation-read observers. A resident cold-source classification is eligibility to initialize, never permission to capture configuration or dispatch.

**Tech Stack:** Python >=3.12, Textual 8.x, existing asyncio/native-read ownership; no dependency or persistence changes.

**Spec:** [Approved Send architecture](../specs/2026-10-06-console-send-preparation-architecture-design.md).

**Implementation task:** [TASK-34563.35](../../../backlog/tasks/task-34563.35%20-%20Receive-Console-Sends-while-stock-skill-trust-is-cold.md).

**Task context:** [TASK-34563](../../../backlog/tasks/task-34563%20-%20Design-Console-Send-preparation-and-I-O-ownership.md) and [TASK-34563.15](../../../backlog/tasks/task-34563.15%20-%20Receive-Console-sends-before-checked-preparation.md). This document does not reopen or mark either prior task complete.

**Status:** Integration owner reviewed the plan and approved Task 1 source/control preparation only. Product implementation waits for the original causal RED and review; all native execution remains integration-owned. No cold-initialization implementation or native qualification is claimed. Source anchors were inspected on the integration worktree through `b0bfd8344a`; verify nearby source drift before editing. Tracked candidate: OPT-69 in the [optimization review list](../../Development/console-optimization-review-list.md).

ADR required: no new ADR.
ADR path: backlog/decisions/222-console-send-preparation-and-io-ownership.md.
Reason: this closes a stock cold-source fallback within ADR-222's existing early receipt, finite native operation and screen-free runtime custody contracts. ADR-126's native authority and physical retirement remain unchanged. It adds neither a new scheduler nor a new permission boundary.

## Global constraints

- Initial receipt remains synchronous and resident-only. No lazy getter, service construction, native read, async task creation for initialization, or import of an absent proof dependency occurs during classification.
- Saved Send failure stops with the draft kept; existing temporary chats remain available. Exact input revisions, attachments, queue authority, hooks, consent, durable acceptance, provider gates and history retain their order.
- Only proven stock cold trust qualifies. Cold registry/consent/persona/database, memory-affine database, custom methods/factory/descriptor, missing proof metadata and unknown source retain their preceding route before receipt.
- Once this route receives an intent, failed or changed initialization refuses that exact intent. It never falls back to synchronous capture after an await.
- No Screen, Widget, bound UI method or frame survives in the preparation source. Navigation alone does not invalidate the received turn.
- Retain source/profile/runtime/controller/store/session and input checks before and after initialization, before native construction, after native return, and before either shared service or configuration publication.
- Physical work stays owned through repeated cancellation. The existing singleflight lock remains held until its actual builder callback returns. Observe the same read handle in both controller and runtime sets.
- Actual rendered/input feedback remains <=100 ms and ordinary application overhead before provider entry remains <1 s. Cold initialization duration stays reported and included in raw whole-Send results; moving it off-loop does not prove either target.
- Targeted checks only. Native execution and timing are sequential under the integration owner. No polling narrowing, cadence change, blanket prewarm, trust bypass, new cache, scheduler or storage mode belongs to this change.

## Evidence and current source

The first original held-configuration run at `557385e36e` fell through `receive_console_visible_intent` and `capture_turn_configuration_snapshot(selection=None)` into the synchronous mounted builder. Passive exception observation identified `console_configuration_preparation._ready_attribute` refusing `LocalSkillsService._trust_service=None` with its factory present. The artificial eight-second probe hold measures neither normal cold construction nor savings.

- `UI/Console_Modules/wiring.py:receive_console_visible_intent` calls the strict stock guard before `runtime.accept_received_intent`; its False result currently precedes receipt.
- `Chat/console_configuration_preparation.py:_source_references` discovers the cold trust slot; `standard_console_configuration_sources` combines readiness and structural compatibility and catches that refusal as False.
- `Chat/console_received_dispatch.py:_run_received_intent_bound` yields once, checks received identity, completes existing hook readiness/review, then calls controller configuration capture. This is the exact insertion point for received cold initialization.
- `app_service_wiring.py:ensure_local_skill_trust_service` already accepts `_source_current`, `_owner_current` and `_read_observers`, builds under its existing async lock, calls `run_preparation_read`, checks ownership before publication, and preserves an app service installed during the build.
- `app_service_wiring.py:_capture_console_skill_trust_preparation` already proves stock app/local/facade/factory/config/profile sources. Its App-context, worker-manager and no-custom-task-factory checks belong to its skill-discovery scheduling adapter.
- `_prepare_console_skill_trust_service` runs that existing adapter in `console-skill-trust-setup`. `SkillsScopeService._call` enforces its existing policy before invoking it. Preserve both contracts.
- `LocalSkillsService.trust_service` returns an existing local winner or publishes the app winner through the original factory. After the app winner is ready, this original getter performs no construction. Do not write its backing field directly.
- `console_preparation_reads.preparation_reads_for` includes `session_id=None` app-scoped reads in session teardown. No new per-session native-read registry is needed.

## Demand decision required before Task 2

The original skill capture deliberately avoids trust for built-in records, and later frozen-definition validation returns before reading trust when no definition digest exists. An unused cold factory therefore must not be initialized merely to make a readiness predicate pass. Task 1 first pins the original cold-receipt failure. Before product edits, review whether an existing finite catalog capture can prove no trust demand for the attempt while retaining stock factory/source custody, or whether another dependent original consumer requires initialization. The finite-initializer integration below is the required-trust branch, not authorization to eagerly initialize every cold Send. Do not infer absence from a UI cache, a filesystem existence shortcut or a previous catalog. Any change to the strict guard/capture contract needed for the no-demand branch must be written and reviewed before Task 2.

## File ownership and exact interfaces

The implementation owner owns these changes as one coherent patch. Other lanes may review or prepare independent controls, but must not edit these production files concurrently.

| File | Change |
| --- | --- |
| `tldw_chatbook/app_service_wiring.py` | Factor the existing service-source proof and preserve the old skill-discovery adapter; keep the original finite initializer. |
| `tldw_chatbook/Chat/console_configuration_preparation.py` | Add received-only classification; share structural reference validation with the unchanged strict guard. |
| `tldw_chatbook/UI/Console_Modules/wiring.py` | Pass classified resident preparation to runtime with the existing intent. |
| `tldw_chatbook/Chat/console_runtime.py` | Carry the optional preparation into the existing received source/custody task. |
| `tldw_chatbook/Chat/console_received_dispatch.py` | Own initialization after existing hook readiness and before checked configuration capture. |
| `Tests/Chat/test_console_cold_trust_preparation.py` | New focused classification and source-change controls. |
| `Tests/UI/test_console_cold_trust_send.py` | New real Send, input/paint, navigation, draft and completion controls. |
| `Tests/Performance/test_console_received_trust_owned.py` | New original native builder/custody controls reusing existing native diagnostics. |

### Shared stock service proof

In `app_service_wiring.py`, introduce private `_ConsoleSkillTrustSource` and:

```python
def _capture_console_skill_trust_source(app, service) -> _ConsoleSkillTrustSource | None: ...
```

The captured source contains only app, runtime, scope/local service, their exact plain field mappings, original factory/descriptor/function metadata, original build lock, resident config/profile identities, loop/thread identity and the validators below. It holds no skill controller, worker, screen, permission verdict or native lease.

Expose only these internal operations on the capture:

```python
source.source_current() -> bool
source.owner_current(expected_app_trust=_INITIAL_TRUST) -> bool
source.publish_local_winner(expected_app_trust) -> object
```

`source_current` is usable from the native callback and verifies original helper code/globals/defaults/closures, module bindings, native profile selectors/cache identities, app/runtime identity and shutdown, local/facade/factory/lock/config/path/policy-collaborator identities. `owner_current` additionally checks the selected event loop/thread and expected app trust slot. Neither invokes a Textual accessor or reuses a permission verdict. The private `_INITIAL_TRUST` sentinel selects the app slot captured at receipt, including an already-ready app winner; an explicit argument validates the winner observed under the original lock. Both permit the intended app trust publication transition only through that expected value. The received wrapper preserves the sentinel when the initializer invokes `_owner_current()` without arguments, so app-ready/local-cold eligibility does not fail on an accidental None default.

`publish_local_winner` runs on the owning loop without an await. Recheck sources and exact original `LocalSkillsService.trust_service` descriptor/factory, read the original property, and recheck sources. Preserve a local winner already installed during the build. If it is still empty, its proven original factory can only return the ready checked app winner. Never invoke the original factory while the app slot is empty; never overwrite either winner. The unchanged full configuration guard below determines whether the resulting live trust source is supported.

Factor the existing 4299–4697 proof by responsibility, not by copying its metadata checks. Keep `_capture_console_skill_trust_preparation(app, service, owner_current, controller_source)` and `_prepare_console_skill_trust_service(captured)` signatures/result shape intact. That existing adapter adds its original controller-source, external owner, App-context, App-worker-manager, original scheduling, and `loop.get_task_factory() is None` requirements around the common proof. Keep its original policy-before-prepare ordering and current cold-only eligibility. Register new helper bodies in the existing definition-time source records; do not add a parallel source-validation framework.

For the received adapter, common eligibility permits a cold local slot with either an empty app slot or an already-ready stock app slot. The latter is partial publication, not a reason to repeat construction. Preserve existing ready/custom discovery behavior. A newly replaced helper, unexpected descriptor or missing source record declines classification without executing it.

### Received-only classification

In `console_configuration_preparation.py`, add frozen private `ConsoleReceivedConfigurationPreparation` with `trust_source: _ConsoleSkillTrustSource | None` and:

```python
def capture_console_received_configuration_preparation(
    app, store, creator, *, session_id: str
) -> ConsoleReceivedConfigurationPreparation | None: ...
```

`trust_source=None` means the original strict ready guard passes. An actual trust source means all other stock configuration owners are ready and compatible and this exact original local trust factory is eligible for finite initialization. `None` means this optimization does not apply.

Extract the structural checks from `standard_console_configuration_sources` into one private `_standard_configuration_references(refs) -> bool`; the existing public strict function keeps its signature, return behavior and full trust-readiness requirement. Factor reference discovery to accept only a captured `_ConsoleSkillTrustSource` through a private `_cold_trust` keyword used by the new classifier. The capture must prove the exact same app/scope/local objects and still-empty local trust slot before reference discovery represents trust as deferred. All remaining references, types, callbacks, databases and affinity checks are identical to the strict path. No boolean `allow_cold`, fake ready service or caller-provided trust override is accepted.

Use already-resident modules and definition-time records to recognize the stock helper. Missing proof dependencies decline the new classification instead of importing/warming them. The deferred reference result cannot be used by `capture_console_turn_configuration_owned`; only received classification sees it. After initialization, call the original strict guard with no deferred argument.

### Existing custody handoff and ordering

Add a private optional keyword to `ConsoleRuntime.accept_received_intent`:

```python
def accept_received_intent(self, intent, *, _configuration_preparation=None) -> str: ...
```

Pass it into the existing `received_preparation_source(runtime, *, configuration_preparation=None)` and `ReceivedPreparationSource.configuration_preparation` field. Existing callers default to their current behavior. The runtime rejects malformed supplied preparation before claiming the draft. The source capture is attached to the existing coroutine/record lifetime and is released with it; no separate task, queue or pending-initializer index is created. Do not put these live source references into the detached input/selection value models.

`receive_console_visible_intent` calls the classifier at its current strict-guard boundary and passes its result only when the remaining original receipt guards pass. It does not call the initializer, publish local trust, or advance durable acceptance.

Add in `console_received_dispatch.py`:

```python
async def _prepare_received_configuration_sources(runtime, record, source) -> None: ...
```

Call it immediately after the final existing hook readiness check and `require_received_source`, immediately before `controller.capture_turn_configuration_snapshot(..., selection=intent.selection)`.

1. No cold preparation: preserve the current path.
2. Cold preparation: call `require_received_source`; validate common sources and current expected app trust, plus runtime/controller/store/read-set identities captured for this operation.
3. Await the original `ServiceWiringMixin.ensure_local_skill_trust_service` with `_source_current` combining pure common-source validation and `require_received_source`, `_owner_current(expected)` combining the same checks with `source.owner_current(expected)`, and `_read_observers=(runtime._preparation_reads, controller._preparation_reads)`.
4. Recheck received/source/owner state. Publish through `publish_local_winner(result)` without overwriting an app or local winner. Recheck the received inputs and both read-set identities.
5. Require `standard_console_configuration_sources(app, store, controller, session_id=record.session_id)` to pass unchanged, then execute original configuration capture. Failed strict validation raises `RecoveryRequired` and retains the draft; never resume the legacy synchronous path.

The validators return literal True only after all checks pass; a raised `RecoveryRequired` propagates. Native checks use only source-safe/read-locked resident state. The owning loop check belongs only to `_owner_current`, not the worker callback. Existing initializer handling of a pre-existing app winner and a winner installed while holding the lock stays intact; before-entry source drift refuses, while a supported winner during the original build is preserved and requalified.

## Review focus

1. A source replacement can occur between receipt and native dispatch, during the original callback, or before local publication. Every interval needs a causal refusal control.
2. Navigation must not cancel app-owned work, while draft edits, Stop, session Close and shutdown must prevent promotion and retain physical retirement.
3. Concurrent cold Sends must use the existing lock and one original constructor; cancellation of the first caller cannot release the lock or its native handle early.
4. An app/local service winner installed during construction must survive, including unsupported winners that force refusal rather than silent overwrite or fallback.
5. Existing skill discovery must retain policy-before-initializer and its App-worker/custom-scheduler compatibility behavior after extracting common proof.

## Task 1: Prove the cold received failure and freeze source classification

**Files:** New Chat/UI/native control modules above; corresponding existing received-intent/native trust tests are fixtures and references, not to be weakened.

**Interfaces:** Consumes original `receive_console_visible_intent`, `accept_received_intent`, `ensure_local_skill_trust_service`, original native builder and the real SQL/configuration hold. Produces original causal failure evidence and source classification cases.

- [ ] First add `test_cold_stock_send_receives_before_original_configuration_sql` for driver Enter and Send button with original file-backed sources and both trust slots proven cold. The existing actual Workspace SQL hold establishes the current fallback even when a built-ins-only profile never asks a trust question. An independent releaser records claim/custody/draft/native-thread state and allows original event-loop progress before releasing within 0.5 seconds. Require receipt before that original native entry and off-loop execution; no manually forced lazy getter or fake ready flag. Qualify the original native hold and cleanup before the causal assertion. This is the first baseline sent for root execution.
- [ ] Add `test_cold_stock_send_receives_before_original_trust_builder` for actual driver Enter and Send button. Use a fresh app, a real managed user skill requiring its original first trust decision, original file-backed profile/services and remote-adapter boundary only. Built-in records deliberately avoid trust and cannot qualify this builder control. Prove app/local trust is cold immediately before Send; never reset a live service to manufacture cold state. Hold the actual original builder callback with passive original-body instrumentation and record its thread, issuing custody task and original callback lifetime. Before release, require current claim, request=None, original draft/inputs intact, no provider entry, original Preparing paint and another accepted/painted keystroke. Record raw input/paint intervals against <=100 ms; a native callback on MainThread or receipt absent at its entry is the original causal RED, separate from harness timeout.
- [ ] Add `test_received_configuration_classifies_only_stock_cold_trust`: ready, both slots cold, app-ready/local-cold, cold unrelated service, memory database, custom lazy factory, modified getter/helper body and absent proof module. Custom/unknown cases must invoke no substituted callback during classification. Ready and cold classification performs zero native reads, service construction or import of absent proof dependencies. New-helper unit failures are not claimed as the original UI causal RED.
- [ ] Add `test_deferred_trust_never_satisfies_strict_configuration_guard`: strict guard remains False while trust is cold; a received-only classification does not make ordinary capture or a provider entry possible.
- [ ] Root runs only these controls sequentially, freezes original source hashes, and records qualification, causal failure and cleanup separately. Do not lengthen a hold to hide main-loop fallback.
- [ ] Review the original failure and the extracted proof boundary before product edits. Commit controls/evidence separately.

## Task 2: Reuse finite initialization under received custody

**Files:** The five production files and classification tests listed above. No changes to `console_preparation_reads.py` or controller capture are expected.

**Interfaces:** Implements exactly the common proof, received preparation value, optional runtime/source handoff and received preparation function specified above. Keeps existing strict guard, received input model, initializer signature and skill-discovery adapter signature intact.

- [ ] Extract common source proof and its definition-time metadata from the current App-worker-specific capture. Retain existing skill-discovery policy/scheduling adapter checks.
- [ ] Implement received-only stock cold classification and shared structural validation, with strict ready semantics unchanged.
- [ ] Carry preparation through existing runtime source ownership and call the finite initializer only after hook readiness inside `_run_received_intent_bound`.
- [ ] Publish the checked original local winner; re-run the unchanged strict guard; use the original configuration capture and durable dispatch pipeline.
- [ ] Run Task 1 controls sequentially and the existing `test_stock_console_skill_setup_has_original_owned_lifetime` and `test_original_skill_trust_builder_retains_native_callback` controls. Report real original-source GREEN separately from classification unit results.
- [ ] Self-review exact source/body/default/closure validation, no retained UI object, no native work during receipt, no synchronous fallback after cold receipt and no wider provider-setting merge. Commit this bounded integration.

## Task 3: Qualify cancellation, races and actual Send completion

**Files:** The new focused control modules; existing native/received tests remain intact. Update the implementation leaf, OPT-69 and this plan with exact results.

**Interfaces:** Consumes the integrated original runtime, App/locals, singleflight lock, read handles, saved turn and actual view. Produces causal ownership/draft evidence and sequential whole-Send measurements.

- [ ] `test_received_trust_cancel_keeps_native_owner_until_return`, with cancel, repeated cancel and callback-raised cancellation: while the original callback is held, its future is running, issuer is unfinished, lock stays held and the same read is visible to runtime/controller drains. After release all exact reads/tasks/locks retire; no provider or draft clear. Existing native error is consumed without an unretrieved exception.
- [ ] `test_received_trust_source_change_refuses_before_publication`, with pre-dispatch, held-native and post-native/pre-publication changes to config module/profile generation or identity, helper body/factory closure, facade/local identity, runtime/controller/store and read-set identities. The relevant check must actually execute; source drift cannot publish local trust, capture configuration or dispatch.
- [ ] `test_received_trust_two_sessions_share_original_initializer`: receive two real session intents against one cold app. Hold original constructor, prove only one constructor executes and the second waits on the existing lock. In the uncancelled case, one constructor serves both intents. If the first issuer is cancelled during construction, preserve the existing initializer's cancellation policy: its un-published result is not repurposed; a waiter may construct after that callback physically retires and releases the lock. Assert no overlapping constructors or early lock release, not one lifetime constructor across cancellation. Neither outcome may clear the other draft or bypass its hook/strict-source checks. No new service registry or retry task.
- [ ] `test_received_trust_preserves_ready_winners`: app winner and local winner during the original callback survive by identity. Supported winners complete checked capture; unsupported winners cause strict refusal with draft preserved. Already-ready app/local-cold state performs no constructor call.
- [ ] `test_received_trust_navigation_keeps_original_turn`: detach/navigate while native construction is held; the exact turn continues after release and does not clear the new view's draft. Inspect newly added fields and closure captures for a retained Screen/Widget/bound UI callback/frame, stopping at the existing app/runtime/controller/service roots; do not recursively traverse the whole App object graph and mistake its established ownership for a new UI continuation. Do not make current view identity an execution gate.
- [ ] `test_received_trust_close_stop_shutdown_retire_before_cleanup`: Stop, close the original session, replace runtime and quit while construction is held. Original owner seals/cancels; no accepted turn/provider follows. Native callback, observer sets and custody must retire before original database/profile creator close. A popped Screen alone is not a cleanup oracle.
- [ ] `test_received_trust_newer_or_retyped_draft_survives`: same-text retyping and changed text invalidate the exact unpromoted intent, preserve the new authored revision and leave no duplicate admission.
- [ ] `test_received_hook_refusal_precedes_cold_trust_build`: original pending/denied hook review prevents constructor entry, configuration capture and provider dispatch; cancellation retains the authored draft. A ready review proceeds through the same checked initialization.
- [ ] `test_cold_received_send_completes_after_checked_initialization`: release the original builder and require original strict guard/capture, saved acceptance, one real adapter entry and completed reply. Inject saved persistence failure at its existing boundary in a separate targeted case; require no adapter and retained draft. Temporary chat retains its existing persistence policy.
- [ ] `test_skill_discovery_policy_and_scheduler_contract_survives`: retain original denied-policy, ready, custom facade, original worker, local/app winner, runtime replacement, shutdown and helper/metadata tamper controls. Eager task factory is supported only by the existing received-custody scheduler; do not relax the discovery adapter's custom-scheduler fallback.
- [ ] Root performs final integrated targeted checks after implementations are saved. Run cold then warm native samples sequentially, reporting raw receipt/input/paint, trust construction, configuration and whole-Send adapter-entry intervals with original I/O attribution. No concurrent native run; no startup or cold cost excluded from a speed claim.
- [ ] Record exact source/HEAD, pass/fail/skip counts, source qualification, original physical retirement and any pending host workers. Do not mark the implementation task Done or OPT-69 implemented until its defined outcomes and review pass. Whole-app responsiveness/performance remains independently qualified.

## Draft failure discovered by the polling control

`warm-repeated-poll-original-red-2` recorded completed Preparing polls with core returns 5 and 5, but failed first because the composer was empty. The subsequent original-writer diagnostic `received-draft-origin-1` attributed this to fixture setup: `_reconcile_console_after_attach -> on_screen_resume -> workspace._reconcile_console_session_with_registry -> _activate_console_session_for_workspace -> create_session -> _activate_session` selected a different session. Original draft sync then loaded that new session's empty draft. The original received session retained all 23 characters and revision 1, `session_inputs_are_current=True`, and request=None. This is not a demonstrated product draft-loss defect or a completed stable-view polling causal RED.

Source correction `d41973538e` selects the workspace through the original registry before mount and waits for original attachment reconciliation within the unchanged 10-second setup bound. It removes direct session workspace mutation and checks workspace/active/visible/composer/draft ownership before yielding the fixture. No manual full sync or weakened draft assertion is introduced. At integrated `4b984d`, `aligned-warm-poll-controls-1` now produces the intended polling RED (repeated core5 versus zero) with the original draft intact, plus one PASS for next real Send; source/custody are clean and zero pending workers remain. Keep this fixture qualification separate from cold trust initialization; new cold controls must also establish exact view ownership and must not silently warm or reset the service they intend to measure.
