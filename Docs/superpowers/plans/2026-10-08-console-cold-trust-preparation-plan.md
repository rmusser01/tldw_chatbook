# Console cold trust preparation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The integration owner controls implementation assignments and all native runs; this planning lane remains source-only.

**Goal:** Let a proven stock Console Send receive its draft immediately when only skill-trust initialization is cold, then determine actual trust demand in one finite catalog read and initialize only when needed, under existing runtime custody.

**Architecture:** Reuse existing source proof, runtime custody and finite read ownership. Read the original visible skill catalog once; a checked builtin-only result needs no trust construction, while managed records request the existing finite initializer before projection. Reuse that attempt's immutable maximum in final configuration; cold eligibility and catalog data never grant execution authority.

**Tech Stack:** Python >=3.12, Textual 8.x, existing asyncio/native-read ownership; no dependency or persistence changes.

**Spec:** [Approved Send architecture](../specs/2026-10-06-console-send-preparation-architecture-design.md).

**Implementation task:** [TASK-34563.35](../../../backlog/tasks/task-34563.35%20-%20Receive-Console-Sends-while-stock-skill-trust-is-cold.md).

**Task context:** [TASK-34563](../../../backlog/tasks/task-34563%20-%20Design-Console-Send-preparation-and-I-O-ownership.md) and [TASK-34563.15](../../../backlog/tasks/task-34563.15%20-%20Receive-Console-sends-before-checked-preparation.md). This document does not reopen or mark either prior task complete.

**Status:** The integration owner qualified original causal RED in `stock-cold-send-owned-baseline-5` after `c653f5445e` fixed the finite Workspace connection lifetime: both real Send routes reach the exact input-thread SQL-before-receipt assertion after all fourteen App creators and every native counter retire. The reviewed one-catalog contract was then assigned for source-only implementation. Candidate `d7d0cc2d71` implements Task 2 and twenty-two isolated contracts in the dedicated cold-trust worktree; static checks pass. Integration review and all runtime/native qualification remain pending under the integration owner. No measured Send/input gain is claimed. Tracked candidate: OPT69 in the [optimization review list](../../Development/console-optimization-review-list.md).

ADR required: no new ADR.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: this closes a stock cold-source fallback within ADR-225's existing early receipt, finite native operation and screen-free runtime custody contracts. ADR-126's native authority and physical retirement remain unchanged. It adds neither a new scheduler nor a new permission boundary.

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

## Demand and compatibility decision

The original `_visible_records` reads package builtins and the current managed index. `_load_index` strips service-assigned keys before overlaying managed records; a same-name managed override cannot claim builtin provenance. `_trust_fields_for_record` and `capture_skill_context_maximum` skip trust for original builtins, and later frozen-definition validation returns before trust when no digest exists. Preserve this existing behavior. Resident source metadata alone cannot establish current catalog contents.

This is a Send-configuration demand contract, not a whole-startup construction guarantee. Original `LocalSkillsService._content_sources` resolves `trust_service` before `get_context`, including builtin-only discovery. The eager separate-host fixture therefore warms synchronously; the genuine stock App/default-worker adapter initializes finitely under its existing owner. Preserve that independent discovery contract. A completed builtin-only Send must add no unused constructor or duplicate initializer, but an already-running original discovery initializer is legitimate. A whole-app zero-constructor assertion cannot prove this narrower requirement.

One finite read of that original catalog determines demand. Do not read the index separately to ask whether it exists, consult a UI cache, initialize for every cold Send, or rescan the catalog after initialization. The existing strict public guard remains strict for ordinary callers. A prepared received-only skill result explicitly permits a known unused cold factory in the owned capture; that private contract is the new branch requiring review before product edits.

## File ownership and exact interfaces

The implementation owner owns these changes as one coherent patch. Other lanes may review or prepare independent controls, but must not edit these production files concurrently.

| File | Change |
| --- | --- |
| `tldw_chatbook/app_service_wiring.py` | Factor the existing service-source proof and preserve the old skill-discovery adapter; keep the original finite initializer. |
| `tldw_chatbook/Chat/console_configuration_preparation.py` | Received-only classification, one owned catalog result and final prepared-skill source validation; share structural checks with the unchanged public strict guard. |
| `tldw_chatbook/Chat/console_configuration_capture.py` | Factor original catalog projection so received preparation can reuse captured records/maximum without a second catalog read. |
| `tldw_chatbook/Chat/console_chat_controller.py` | Carry the private prepared-skill result through existing selected configuration capture. |
| `tldw_chatbook/Skills_Interop/local_skills_service.py` | Extend existing definition-time source records for the original catalog/projection methods relied on by demand classification; no catalog behavior change. |
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

`publish_local_winner` is used only after actual managed-record demand. It runs on the owning loop without an await. Recheck sources and exact original `LocalSkillsService.trust_service` descriptor/factory, read the original property, and recheck sources. Preserve a local winner already installed during the build. If it is still empty, its proven original factory can only return the ready checked app winner. Never invoke the original factory while the app slot is empty; never overwrite either winner. The unchanged full configuration guard below determines whether the resulting live trust source is supported.

Factor the existing 4299–4697 proof by responsibility, not by copying its metadata checks. Keep `_capture_console_skill_trust_preparation(app, service, owner_current, controller_source)` and `_prepare_console_skill_trust_service(captured)` signatures/result shape intact. That existing adapter adds its original controller-source, external owner, App-context, App-worker-manager, original scheduling, and `loop.get_task_factory() is None` requirements around the common proof. Keep its original policy-before-prepare ordering and current cold-only eligibility. Register new helper bodies in the existing definition-time source records; do not add a parallel source-validation framework.

For the received adapter, common eligibility permits a cold local slot with either an empty app slot or an already-ready stock app slot. The latter is partial publication, not a reason to repeat construction. Preserve existing ready/custom discovery behavior. A newly replaced helper, unexpected descriptor or missing source record declines classification without executing it.

### Resident classification and one finite catalog result

Use the reviewed `ConsoleReceivedConfigurationPreparation(trust_source=None)` and:

```python
def capture_console_received_configuration_preparation(
    app, store, creator, *, session_id: str
) -> ConsoleReceivedConfigurationPreparation | None: ...
```

None declines the new route. A value with no trust source preserves the current ready route. A cold source proves the exact stock factory and all other configuration sources, without reading the catalog or invoking a getter. Extract shared `_standard_configuration_references(refs) -> bool`; the public `standard_console_configuration_sources(...)` signature/strict behavior remains unchanged. Only private received classification may discover references with a proven `_unused_trust` source. Missing proof modules decline without imports.

In `console_configuration_preparation.py`, add frozen operation-local `ConsoleSkillCatalogRead` and `ConsolePreparedSkillContext`. Both bind the exact existing creator/session/turn, profile/config generation, original local/facade/plugin/factory objects and method records. These are result values, not admission owners or reusable caches. The read contains detached records plus either an already-projected maximum or an explicit requires-trust outcome. The final result contains a deeply immutable maximum, its captured source checks and whether trust was used. Its live source references never enter `ConsoleTurnConfigurationSnapshot` or persisted data.

```python
async def capture_console_skill_catalog_owned(
    app, store, creator, *, session_id, turn_id, skill_workspace_id, preparation,
    reads, observers=(), require_current
) -> ConsoleSkillCatalogRead: ...

async def finish_console_skill_catalog_owned(
    catalog, *, reads, observers=(), require_current
) -> ConsolePreparedSkillContext: ...
```

The explicit `skill_workspace_id` retains the existing caller's skill scope (including mounted None versus a runtime workspace); do not silently substitute `session.workspace_id`. Bind it into the result and carry it unchanged into original plugin projection.

Both reuse `run_preparation_read`. Extend the existing source records for original `_visible_records`, `_load_index`, `_summary_for_record`, `_trust_fields_for_record` and their existing provenance helpers; reject custom instance/class/body bindings before consuming them. Reuse existing metadata validation, not a new general checker.

The first native operation calls original `_visible_records` exactly once. Keep its normalization/managed-override behavior. Before calling any summary that could read trust, determine whether any normalized record has source other than builtin. If none, project original builtin summaries and original published-plugin maximum in that same finite operation and return the ready maximum; no initializer or second native catalog operation. If managed records exist, return detached records requiring trust, with no summary/getter call yet.

Factor the existing projection loop into private `_capture_skill_context_from_records(local, records, workspace_id, *, _plugin_service)` in `console_configuration_capture.py`. The existing `capture_skill_context_maximum` keeps its signature and optional-error behavior and delegates to it after its usual original enumeration. The prepared route calls that same projection on its detached records after needed trust initialization; it does not call `_visible_records` again. Original summaries, trust status/digests, builtin verification, duplicate/plugin precedence and nonce generation remain in that projection.

An optional catalog/projection error is recorded as unavailable and yields the existing empty maximum, never as proven catalog absence. Preserve the existing optional empty-maximum policy when sources remain valid. Cancellation, failed current-source/native ownership validation and unsupported service winners propagate as refusal with draft retained. Do not catch those failures inside an optional-data fallback. If an initializer's error cannot be distinguished from failed required source ownership under the existing checks, refuse; do not invent successful no-demand evidence. Tests must pin optional failure versus source invalidation separately.

### Existing custody handoff and final capture

Keep `ConsoleRuntime.accept_received_intent(intent, *, _configuration_preparation=None)` and `ReceivedPreparationSource.configuration_preparation`. Original callers default to their current behavior. The runtime validates any supplied preparation before claiming the draft; no new task, registry or index is created. The screen only passes resident classification with the existing detached intent.

The preparation sequence is inlined in the existing `_run_received_intent_bound` driver after the final hook readiness/source check and before selected configuration capture; it returns a `ConsolePreparedSkillContext` only for the cold branch. This avoids a new one-use wrapper:

1. Ready preparation returns None and retains today's ready capture path.
2. Cold preparation validates `require_received_source`, common source proof and exact read-set identities, then awaits `capture_console_skill_catalog_owned` under the existing controller/runtime read observers.
3. Ready/no-demand catalog proceeds directly to the final result. It must not call ensure or either trust getter.
4. Requires-trust catalog invokes original `ensure_local_skill_trust_service` with source/current callbacks and both observer sets. Worker callback validation is thread-safe; owning-loop checks stay in `_owner_current`. Preserve captured-initial sentinel semantics for app-ready/local-cold, the original singleflight lock, physical retirement and ready winners. Publish the checked original local winner, rerun the unchanged public strict guard, then finish projection over the same records in one finite operation.
5. After each await, revalidate source/received input/owner/read sets. Return the immutable prepared maximum to existing configuration capture. A failed owned route never falls back to synchronous preparation after receipt.

Add private `_prepared_skills=None` to controller `capture_turn_configuration_snapshot` and `capture_console_turn_configuration_owned`. Only an exact selected received attempt may supply this result; custom/unselected callbacks retain their existing route. The owned producer's ordinary case still uses the unchanged strict guard. For prepared no-demand/unavailable optional skills, private reference capture accepts a still-proven stock cold source only together with its exact finite result and original method/source checks; every other structural/database/permission check is identical. Recheck that source/result pair in its original `current()` callback before and after dependent native work. Any factory/local/profile/creator/attempt mismatch refuses rather than reconstructing or reusing the result.

In `capture_console_turn_configuration`, add private `_skill_context_maximum=_UNSET_SKILL_CONTEXT`. Default delegates to original `capture_skill_context_maximum`. An internally supplied deeply immutable prepared maximum is consumed directly, without a second catalog enumeration, new nonce or summary recomputation. The owned caller validates provenance; the snapshot retains only detached data. A managed skill added after the attempt's catalog capture is absent from that attempt's ceiling; existing execution-time authority and builtin pin/digest checks remain live. No permission verdict or source-validity claim survives to a later attempt.

## Review focus

1. A source replacement can occur between receipt and native dispatch, during the original callback, or before local publication. Every interval needs a causal refusal control.
2. Navigation must not cancel app-owned work, while draft edits, Stop, session Close and shutdown must prevent promotion and retain physical retirement.
3. Concurrent cold Sends must use the existing lock and one original constructor; cancellation of the first caller cannot release the lock or its native handle early.
4. An app/local service winner installed during construction must survive, including unsupported winners that force refusal rather than silent overwrite or fallback.
5. Managed same-name overrides and malformed/failed catalogs must not masquerade as no-demand; original skill-discovery policy/scheduling and optional-data error policy remain intact.
6. **Open contract review:** original startup discovery can publish checked ready app/local winners while a no-demand received catalog read is running. The result must permit that legitimate transition through a freshly checked ready source (or an explicitly reviewed original-winner transition), preserve the one captured catalog, and reject arbitrary replacements. An unconditional expected-initial-None check would falsely refuse a valid first Send. This transition must be specified and tested before product implementation; do not relax identity validation globally.

## Task 1: Prove the cold received failure and freeze source classification

**Files:** New Chat/UI/native control modules above; corresponding existing received-intent/native trust tests are fixtures and references, not to be weakened.

**Interfaces:** Consumes original `receive_console_visible_intent`, `accept_received_intent`, `ensure_local_skill_trust_service`, original native builder and the real SQL/configuration hold. Produces original causal failure evidence and source classification cases.

- [ ] First add `test_cold_stock_send_receives_before_original_configuration_sql` for driver Enter and Send button with original file-backed sources and both trust slots proven cold. The existing actual Workspace SQL hold establishes the current fallback even when a built-ins-only profile never asks a trust question. An independent releaser records claim/custody/draft/native-thread state and allows original event-loop progress before releasing within 0.5 seconds. Require receipt before that original native entry and off-loop execution; no manually forced lazy getter or fake ready flag. Qualify the original native hold and cleanup before the causal assertion. This is the first baseline sent for root execution.
- [ ] Add `test_cold_stock_send_receives_before_original_trust_builder` for actual driver Enter and Send button. Use a fresh app, a real managed user skill requiring its original first trust decision, original file-backed profile/services and remote-adapter boundary only. Built-in records deliberately avoid trust and cannot qualify this builder control. Prove app/local trust is cold immediately before Send; never reset a live service to manufacture cold state. Hold the actual original builder callback with passive original-body instrumentation and record its thread, issuing custody task and original callback lifetime. Before release, require current claim, request=None, original draft/inputs intact, no provider entry, original Preparing paint and another accepted/painted keystroke. Record raw input/paint intervals against <=100 ms; a native callback on MainThread or receipt absent at its entry is the original causal RED, separate from harness timeout.
- [ ] Add `test_received_configuration_classifies_only_stock_cold_trust`: ready, both slots cold, app-ready/local-cold, cold unrelated service, memory database, custom lazy factory, modified getter/helper body and absent proof module. Custom/unknown cases must invoke no substituted callback during classification. Ready and cold classification performs zero native reads, service construction or import of absent proof dependencies. New-helper unit failures are not claimed as the original UI causal RED.
- [ ] Replace the unqualified eager-host `test_cold_builtin_only_send_does_not_construct_unused_trust` oracle with a Send-scoped control: actual saved reply, one adapter call, nonempty builtin-only original captured maximum, and no Send-induced construction/duplicate initializer. An original startup discovery initializer may already be running and complete normally. Pin its original task/read before Send, then attribute every additional construction to its original owner; never suppress discovery or reset its slots. Preserve the old setup failures in evidence, and retire the misleading default eager controls only after the authentic replacement qualifies.
- [ ] Add `test_deferred_trust_never_satisfies_strict_configuration_guard`: strict guard remains False while trust is cold; a received-only classification does not make ordinary capture or a provider entry possible.
- [ ] Root runs only these controls sequentially, freezes original source hashes, and records qualification, causal failure and cleanup separately. Do not lengthen a hold to hide main-loop fallback.
- [ ] Review the original failure and the extracted proof boundary before product edits. Commit controls/evidence separately.

## Task 2: Reuse finite initialization under received custody

**Files:** The eight production files and classification tests listed above; changes belong to one cold received-preparation contract. No changes to `console_preparation_reads.py` or its lifetime contract are expected.

**Interfaces:** Implements the common proof, resident classification, one finite catalog read/result, optional runtime/source handoff and private prepared-skill reuse specified above. Keeps existing strict guard, received input model, initializer signature and skill-discovery adapter signature intact.

- [x] Extract common source proof and its definition-time metadata from the current App-worker-specific capture. Retain existing skill-discovery policy/scheduling adapter checks.
- [x] Implement received-only stock cold classification and shared structural validation, with strict ready semantics unchanged.
- [x] Carry preparation through existing runtime source ownership. After hook readiness, capture original records once; initialize only on managed-record demand; project through the factored original loop without rescanning.
- [x] Required-trust branch publishes the checked original local winner and reruns the unchanged strict guard. The no-demand branch uses the explicit finite result/source contract. Both reuse the prepared maximum in original configuration and retain original durable dispatch.
- [ ] Run Task 1 controls sequentially and the existing `test_stock_console_skill_setup_has_original_owned_lifetime` and `test_original_skill_trust_builder_retains_native_callback` controls. Report real original-source GREEN separately from classification unit results.
- [x] Self-review exact source/body/default/closure validation, no retained UI object, no native work during receipt, no synchronous fallback after cold receipt and no wider provider-setting merge. Commit this bounded integration.

## Task 3: Qualify cancellation, races and actual Send completion

**Files:** The new focused control modules; existing native/received tests remain intact. Update the implementation leaf, OPT-69 and this plan with exact results.

**Interfaces:** Consumes the integrated original runtime, App/locals, singleflight lock, read handles, saved turn and actual view. Produces causal ownership/draft evidence and sequential whole-Send measurements.

- [ ] `test_received_trust_cancel_keeps_native_owner_until_return`, with cancel, repeated cancel and callback-raised cancellation: while the original callback is held, its future is running, issuer is unfinished, lock stays held and the same read is visible to runtime/controller drains. After release all exact reads/tasks/locks retire; no provider or draft clear. Existing native error is consumed without an unretrieved exception.
- [ ] `test_received_catalog_reused_once`: count original enumeration/projection for builtin-only and managed branches; managed same-name override requires trust, raw service-assigned tags cannot forge builtin provenance, and a new attempt observes a changed catalog. No second index scan after initialization.
- [ ] `test_optional_catalog_failure_is_not_absence`: preserve an explicit unavailable empty maximum only with valid source; source/helper/profile replacement refuses and keeps the draft. A no-demand result from another attempt/creator/session is rejected.
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

## Cold-control qualification history

Source `e8b58c9f76` prepared the two original SQL receipt controls. The integration run `cold-sql-receipt-baseline-1` at `be8115f315` failed both before Send: original app trust was already initialized when the aligned fixture yielded. Pytest24.337s/driver89.109s, frozen source/HEAD, zero after-fixture workers and normal diagnostic cleanup. This is a setup failure, not the requested causal RED. `e448a7a97a` adds the positive builtin-only original Send control; its cold setup also remains unqualified. `fc420631f1` observes original builder/ensure/skill-preparation ancestry from before app creation to attribute the warming path without replacing it. Do not reset a live service or weaken initial source/cold assertions; establish an authentic cold phase before claiming either result.

`cold-trust-origin-1` at integrated `82b2bb1685` supplies the actual warming ancestry: MainThread `_build_local_skill_trust_service` ← lazy property/factory ← `LocalSkillsService.trust_service` ← `_content_sources` ← `local_content_lifetime.selected/admitted/asynchronous` ← `SkillsScopeService._call/get_context` ← `_fetch_console_skill_context/_refresh_console_skill_candidates` ← Textual Worker/asyncio eager task creation. Builder begins 10.3534304s after observer installation; no ensure or stock preparer entry is observed. Sources are frozen, no after-fixture workers remain, and the driver retires normally in 22.235s. This qualifies the eager-host setup explanation only.

The reviewable replacement reuses the integration owner's `Tests/Performance/test_console_skill_trust_stock_owned.py`: real original TldwCli, default task factory, original constructor-owner retirement and exact App-worker/native callback correspondence. Source-only helper `414e6ec491`, `Tests/Performance/_stock_cold_send_control.py`, selects a real registry workspace before mount and adds driver Enter/button observations while the original discovery builder is held before publication. Its independent SQL releaser is bounded to 0.5s; both holds release before cleanup, and causal receipt assertions run only after the original fixture's App/native/creator retirement. The original ten-second admitted-control bound remains. Helper and proposed integration script compile and helper Ruff passes, source-only; the integration owner subsequently reviewed and executed it as frozen local source. No product source is changed.

`stock-cold-send-original-1` at integration `116628850f` plus frozen helper/fixture changes reaches both authentic routes: configuration SQL is on the input thread before receipt, the loop and original Preparing frame do not progress while held, cold slots/factory/draft/session remain exact, the native lease is live with no timeout, and remote calls remain zero. The original startup builder Future/singleflight retire with invalid=[]; fixture-wide retirement fails afterward. `stock-cold-send-retirement-origin-2` identifies a remaining `db.workspaces` native cache on `asyncio_11`, nontransactional and still leased. The original main-thread close cannot retire that thread-local cache. Getter/caller attribution is pending in the integration lane; no broader closer or relaxed drain is authorized by this observation. Retain the causal blocking facts but do not mark the whole control qualified or begin product implementation on a false cleanup pass.

The subsequent exact-handle trace `stock-cold-send-workspace-origin-3` identifies the remaining worker cache's original birth in `ServiceWiringMixin._compose_tool_pack_service_off_thread` → `ToolPackService.reconcile_receipts` → `_WorkspaceReferences.capture` → `LocalWorkspaceRegistryService.list_workspaces` → Workspace transaction/connection, on its original Textual/concurrent-futures worker (`asyncio_7` in that run). Every other observed Workspace creator handle retired. This is a startup composition ownership leak, separate from Send/helper cancellation; its integration owner is correcting the finite worker. No broader closer, relaxed drain or cold product implementation follows from the diagnostic alone.
