# Shared Console tool preparation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The user subsequently authorized three parallel lanes: shared preparation, controller/provider integration, and baseline verification. Each lane owns distinct files; root is the sole integration owner. Run final integrated checks after both implementations are ready and keep native timing runs sequential. Every task requires its own verification and commit.

**Goal:** Remove duplicate permission loads and worker handoffs from ordinary MCP/local provider composition while preserving enabled tools, source ownership, policy effects and live invocation gates.

**Architecture:** One source-owned preparation operation supplies both stock consumers with detached tool/policy data and a shared compose-time switch observation. Pure catalog adoption does not perform native reads. Initial turn maxima and actual invocation gates retain their separate fresh observations.

**Tech stack:** Existing Python 3.12, Textual 8.x, asyncio, stdlib dataclasses/JSON, existing native source/admission APIs; no new dependency.

**Spec:** [Console Send preparation architecture](../specs/2026-10-06-console-send-preparation-architecture-design.md).

**Plan task:** [TASK-34563.1](../../../backlog/tasks/task-34563.1%20-%20Plan-shared-Console-tool-preparation.md).

ADR required: yes, existing decision applies.
ADR path: backlog/decisions/222-console-send-preparation-and-io-ownership.md; preserve ADR-126's actual source/canonical selection and retirement rules.
Reason: same-operation data sharing is the reviewed first domain slice, not a new lifetime or storage format.

## Scope and execution base

Base: committed integration 6e61cf1da1e4fb14cff80ac69c76c5208dc0550b plus the design-only commits on codex/console-send-preparation-plan. The earlier unproven raw-member batching candidate remains in its original worktree and is not included. Check relevant committed/source drift before each task; do not copy another chat's uncommitted controller changes or wait for unrelated work to finish.

This first implementation plan covers the tool domain only. It can ship independently under existing receipt/custody/durable ordering. Early received-intent custody, runtime initial-hook review and broader lifecycle changes have separate enablement prerequisites in the spec and follow in later plans. Do not insert an early-custody path here or rewrite history/shutdown merely to deliver this optimization.

Measured current composition performs a permission load for MCP's switch, another for effective states, and another for local tools' switch. Ordinary no-mutation stock composition targets one permission observation for these consumers. Initial maximum capture and later invocation are outside that count. Changed-definition writes and their required fresh reads are separately counted; do not force every route into a one-read assertion.

## Global constraints

- Rendered receipt/input target: 100 ms; ordinary application overhead to actual adapter: under one second. This slice must report the remaining gap if it does not meet them.
- Saved-turn commit failure stops Send and keeps the draft; existing temporary chats stay available. Keep WAL/NORMAL and separate execution/file durability policies unchanged.
- Keep every enabled tool, ceiling, definition hash, profile/persona floor, approval, plugin/custom fallback, naming/order/schema and dispatch route.
- No native proof crosses an await. Issued native work drains through repeated cancellation before physical retirement. Existing loop callbacks remain on their loop.
- Shared data does not authorize execution. Invocation/launch/credential/workspace gates remain live. No permission TTL/cache, public fast mode, generic scheduler or duplicated state ledger.
- Live composition can publish inspector counts; disposable preview cannot mutate live presentation. They are separate preparation operations.
- Run targeted tests only. Preserve current deadlines/assertions; full sweeps remain opt-in.

## Review focus

1. The MCP maximum is empty while local tools remain enabled: do not skip their required switch observation or load an unused external catalog.
2. A named profile inherits default policy and persona narrows it: use the same selected profile and floors, without deriving authority from display state.
3. A custom reader/factory or replaced source callback changes semantics: retain its ordinary call shape, affinity and fresh route.
4. Policy/definition mutation or cancellation occurs during preparation: preserve required effects and native drain; reject obsolete ownership publication.
5. A shared schema is nested/mutable or supplied to another provider/source: no alias mutation, forged authority or result reuse across an owning operation.

## Files and responsibilities

- Create tldw_chatbook/MCP/console_tool_preparation.py: domain result and preparation orchestration; reuse captured-source/native helpers from console_snapshot. Do not move public compatibility helpers wholesale.
- Modify tldw_chatbook/MCP/unified_control_plane_service.py: extract pure loaded-payload state resolution; preserve the public method's fresh load and audit contract.
- Modify tldw_chatbook/Agents/mcp_tool_provider.py: adopt one already-prepared stock catalog through an explicit internal method; ordinary public composition remains available.
- Modify tldw_chatbook/Chat/console_chat_controller.py: own one preparation result for stock MCP/local composition and recheck the actual session/service/context before adoption.
- Create Tests/MCP/test_console_tool_preparation.py and Tests/Chat/test_console_shared_tool_preparation.py: real source/read counts, parity, ownership and conditional consumers.
- Reuse Tests/MCP/test_external_catalog_worker_ownership.py, test_console_snapshot_source_contracts.py and Tests/Chat/test_console_async_mcp_snapshot.py fixtures/probes.

## Execution preflight before Task 1

Verify the exact checkout/branch and initial clean staged/working tree. Establish a checkout-specific Python 3.12 interpreter and the existing private-root/contained test runner before any fixture import or test execution; this is a prerequisite, not work deferred until qualification. Confirm the package, Tests and profile_core resolve from this checkout. Run the existing snapshot/catalog controls on unchanged source and record baseline failures and original composition counts. No new code is written until that evidence and its private/native ownership are understood. The user reviewed this plan and authorized execution before implementation began.

## Task 1: Source-owned preparation result and one ordinary policy observation

**Interfaces produced:**

- PreparedConsoleTool, a frozen data row with HubTool's server_key/server_label/source/name/description/tags/stale/executable fields; schema_json: str | None preserves schema key order and None behavior; effective: EffectiveToolState. to_hub_tool() -> HubTool returns a run-owned schema mapping, not shared mutable state.
- ConsoleToolPreparation, with profile_id: str, kill_switch: bool, tools: tuple[PreparedConsoleTool, ...] and private source-owned provenance. require_current(service: UnifiedMCPControlPlaneService) -> None checks applicability at adoption, not at every pure field read. Provenance grants no permission and retains no native lease.
- async prepare_console_tools(service: UnifiedMCPControlPlaneService, *, profile_id: str = 'default', include_mcp_catalog: bool, need_local_switch: bool, builtin_raw_name_exclusions: frozenset[str], owned_profile_ids: frozenset[str]) -> ConsoleToolPreparation | None. None means unsupported stock sharing: callers use their existing ordinary route. Native/owner/cancellation errors retain their actual failure, never a fabricated empty success. If neither consumer participates, decline before constructing captured sources; the original empty/disabled route remains zero-I/O.
- _resolve_tool_states_from_payload(payload: dict[str, Any], tools: Sequence[HubTool], *, profile_id: str) -> dict[tuple[str, str], EffectiveToolState] in the control-plane service module is pure and calls the existing resolve_effective_state.

- [ ] Write failing tests using the real snapshot_case/counted original MCPPermissionStore.load and catalog reader. Pin ordinary load count 1; killed state loads policy once and skips catalog/inventory; no participating consumer performs zero source reads and creates no provider; local-only performs one switch observation and zero external catalog loads. Test source change, repeated cancellation, schema aliasing, nested values and named profile parity.
- [ ] Run only the new test file and confirm meaningful failures on current code. A missing new API alone is not the final RED evidence: also characterize the current three-load composition count in Task 2's old entry point.
- [ ] Implement the minimal domain result and preparation API using existing _CapturedSources, _checked_read and _owned_worker. Resolve loop-owned inventory/profile callbacks on their documented loop. Use one loaded permission payload for ordinary projection; preserve the exact catalog source and early killed behavior. No worker callback invokes arbitrary loop-only extensions.
- [ ] Extract the pure resolver in unified_control_plane_service without changing effective_tool_states' public signature, fresh load, missing-store behavior or downgrade audit. The new owner preserves required downgrade/emit-once behavior through existing native methods; changed/effectful paths may need additional qualified fresh reads. Do not save an old payload or drop the source/canonical checks to satisfy the normal-path count.
- [ ] Preserve original reader/helper qualification. Sharing declines for unsupported/custom sources or replaced dependencies, keeping their ordinary method order and affinity. Normalize/freeze once, with existing domain limits; unsupported non-JSON schema data must retain the documented ordinary route, not be silently truncated.
- [ ] Run Tests/MCP/test_console_tool_preparation.py plus the directly affected existing control-plane/permission and external-catalog controls; check original outcomes, worker threads, native lease retirement and audit counts. Run scoped Ruff/format and diff checks; fix only new findings.
- [ ] Commit the independently verified domain API and controls.

## Task 2: Route stock MCP and local composition through one result

**Consumes:** Task 1's prepare_console_tools and ConsoleToolPreparation.

**Interfaces produced:** MCPToolProvider.adopt_console_preparation(preparation: ConsoleToolPreparation, *, _controller_composition: _ControllerCatalogComposition) -> None, synchronous on the owning loop. It validates source/operation applicability and the selected profile against this provider once, clears old stamped decisions, applies the existing exclusions/maxima/definition hashes, deduplicates names over the complete set, and installs run-owned HubTools/catalog entries and disconnected counts. It makes no native read and does not alter invoke().

Keep compose_catalog() and all existing controller wrappers' supported signatures. Add private stock-only wiring where needed; never pass new arguments to a custom factory/reader that did not opt into this contract.

- [ ] Extend the real controller source fixture to call the original _compose_agent_request_providers. Baseline RED asserts one permission load instead of the existing three while checking identical catalog IDs/order/descriptions/schemas, local exposure and inspector outcome. Use the original _MaximumProbe/ReadProbe pattern; do not mock away storage, source guards or the preparation owner.
- [ ] Add cases for empty/unset/nonempty maxima, local disabled/local-only, switch enabled, named/default profiles, persona narrowing, disconnected counts, plugin maximum (including empty mapping), custom factories/readers, preview publication and changed service/session/root/context across awaits. New pure consumers must add zero native reads/admissions.
- [ ] Implement adopt_console_preparation by reusing the current pure naming/filter/install logic. Keep catalog clearing and decision-stamp retirement. to_hub_tool creates provider-owned mutable schemas where the existing API expects dicts, preserving definition hashes and key order.
- [ ] In _compose_agent_request_providers, determine actual participating stock consumers and capture their selected profile and original source/context. Request one preparation result. Recheck applicability before each publication/adoption; pass the same switch data to the existing local builder through its qualified stock seam without another asynchronous switch read.
- [ ] Keep standalone/direct/custom/plugin composition on its current route unless its explicit contract is qualified. An empty MCP maximum cannot suppress a needed local switch read. Changed-definition effects and unsupported callback fallbacks retain their original native semantics and counts, documented separately from the ordinary one-load case.
- [ ] Preserve fresh actual invocation checks. Add a test flipping the real switch after composition and before the real gate: cached compose-time off cannot execute a now-blocked call. Preparation failures are not converted to an off observation, and this slice never broadens an earlier approval. Existing invocation-wrapper error-policy hardening is explicitly a later security/lifecycle slice; do not claim that the current legacy invocation code already implements it.
- [ ] Run new integration controls and existing Tests/MCP/test_console_snapshot_source_contracts.py, Tests/Chat/test_console_async_mcp_snapshot.py and relevant local/MCP provider/preview controls. Keep exact physical drain, source-replacement and original assertions. Scope lint/format to modified files and compare pre-existing findings.
- [ ] Commit the verified stock integration and parity/count evidence.

## Task 3: Qualify actual latency, source and retirement

**Files:** existing Tests/Performance/test_console_native_pause_probe.py and native ownership helpers; diagnostic-only changes, if necessary, belong beside those existing helpers. Do not copy the old heavy observer or change product budgets to manufacture a pass.

- [ ] Reverify the preflight interpreter, checkout imports and private-root/containment identity before full-app diagnostic runs. Reuse valid installed dependencies; no shared-profile launch, broad package reinstall or live provider credential is required for the immediate adapter diagnostic.
- [ ] Run the directly affected native/JSON fixture baseline on this base first. Attribute every failure against exact unchanged source; do not relabel Windows/private-profile fixture failures as regression acceptance or extend their deadlines. No full sweep.
- [ ] Record original load/admission/write/worker counts by operation and path shape for unchanged baseline and candidate. Confirm normal composition permission loads 3 -> 1 with identical tool outputs, and no extra native work from a pure consumer. Count changed/audited/fallback paths independently; preserve per-target/canonical admission.
- [ ] Use the existing Windows-contained private full-app runner and its normal physical process retirement. Match interpreter, dependencies, config/feature shape, filesystem placement and minimal instrumentation. Alternate two baseline/candidate pairs with three sends each; retain cold/warm and streaming/non-streaming samples individually. Only final provider I/O may be immediate.
- [ ] Drive real Enter/button paths and observe actual rendered receipt and input responsiveness while an original native read is held. If existing UI feedback still misses 100 ms, report it and carry the spec's receipt/approval prerequisite plan forward; do not claim this tool slice fixed it from a status assignment or direct action call.
- [ ] Require real replies, complete trace links, correct checkpoint settlement, current source, no diagnostic overflow and positive native/pump/process retirement. Report unadjusted Send-to-adapter samples and every remaining gap to the one-second goal; overlapping timings are not an additive savings estimate.
- [ ] Qualify Linux and macOS separately through the repository's existing targeted host workflow after the candidate is reviewable. Missing host access is missing evidence, not a positive result. Preserve original native 15-second regression assertion as well as the stronger product targets.
- [ ] Self-review source/API parity, permissions, privacy, performance and scope; update actual Backlog task AC/notes and ADR/spec links. Commit diagnostic/report changes only after evidence is reviewed. Do not mark overall Send stability/speed complete merely because counts improve.

## Planning self-review and follow-on boundaries

This first plan deliberately removes a demonstrated duplicate read chain under current custody and durability. It does not attempt early received-intent promotion, a new global approval state, background history persistence, database synchronization changes or storage-format migration. These contracts remain in the accepted architecture and receive separate dependency-ordered plans after measured results.

The implementation plan was reviewed before product changes. The latest user instruction selects the three parallel lanes described above. Delivery tasks are TASK-34563.2, TASK-34563.3 and TASK-34563.4; each was created, planned and read before its implementation. Do not reopen the execution-method decision.


## Exact scoped commands and plan checks

Run pytest commands only from the verified checkout-specific private-root/contained runner established in Execution preflight, after its root and imported package paths are checked. That setup is mandatory even for these targeted commands; do not fall back to the real user profile.

```powershell
.\.venv\Scripts\python.exe -X utf8 -I -m pytest Tests/MCP/test_console_tool_preparation.py -q -rs
.\.venv\Scripts\python.exe -X utf8 -I -m pytest Tests/Chat/test_console_shared_tool_preparation.py Tests/MCP/test_console_snapshot_source_contracts.py Tests/Chat/test_console_async_mcp_snapshot.py -q -rs
.\.venv\Scripts\python.exe -X utf8 -I -m pytest Tests/MCP/test_external_catalog_worker_ownership.py Tests/MCP/test_external_catalog_single_load.py -q -rs
```

Expected acceptance is no new candidate-only failures, unchanged original assertions, expected native counts and positive retirement. A raw pytest failure is attributed rather than waived; known baseline failures remain explicitly outside acceptance. New mutation/ownership controls must fail when their corresponding behavior is deliberately removed.

Run the configured Ruff binary with --no-cache on the modified Python files and format-check newly created modules. Compare pre-existing lint/format findings on existing large modules; do not reformat unrelated source or raise size/boot ratchets. `git diff --check` must succeed before every commit.

Self-review: this plan has one shared type/API consumed by the integration, no undeclared helper or future task reference, and concrete negative cases for every Review Focus line. Runtime and performance checks are planned requirements, not claimed passing results. The one-load normal observation does not encompass mutations, initial maxima, resumed approval or execution. Full-source compatibility and actual latency remain independent of count improvement.


## Execution result

The shared preparation and controller/provider lanes are implemented and reviewed. Root ran the final integrated checks after both lanes were ready: 187 targeted checks passed. All four matched native timing runs ran sequentially. The full-app count diagnostic confirms one permission load in each of three stock compositions.

Latency acceptance remains open: candidate Send-to-adapter samples span 7.374–11.669 seconds, the original native regression is red, actual 100 ms terminal feedback is unqualified, and candidate Linux/macOS native checks are missing. Unchecked qualification items above remain requirements, not inferred passes. See [the verification report](../../Development/2026-10-06-console-shared-preparation-verification.md) for results and limitations. Tasks remain In Progress.


## Constructor provenance correction (2026-10-07)

ADR required: yes, existing ADR-222 applies.
ADR path: backlog/decisions/222-console-send-preparation-and-io-ownership.md.
Reason: repairs the already agreed stock/custom preparation boundary; no new architecture or permission authority.

Independent review found that callable-body checks omit in-place constructor and field-lookup changes on the shared result classes and the HubTool constructor used by conversion. Verify with bounded real substitutions before changing product code. A changed stock dependency must decline sharing before native reads, and an issued result must refuse adoption before invoking the replacement. The ordinary custom route remains available; this is not general Python tamper protection.

1. Verification lane owns only Tests/MCP/test_console_preparation_constructors.py. Cover directly consumed allocation, lookup and conversion slots, including replacement before first preparation import. Keep allocator mutation isolated from later cases and normal fixture cleanup.
2. Shared lane owns MCP/console_tool_preparation.py and the minimal defining-module HubTool anchor in MCP/hub_tool_catalog.py. Reuse static, definition-time qualification and existing failure paths; add no native work, cache or generic guard framework.
3. Root integrates after reviewing the plan and observing meaningful REDs. Review all consumed slots, then run focused constructor, preparation and controller/provider controls through the existing contained runner. Native runs are sequential and coordinated with UAT.
4. Compare scoped lint/format to the committed base, record actual normal retirement and source state, update task 34563.2 and commit the correction. Resume task 34563.9 after this narrow repair. No latency or cross-host claim follows from these controls.

Constructor RED evidence: on unchanged product 0a0582, `preparation-constructors-red` failed all 13 new allocation/lookup/conversion controls at their actual boundary assertions (including the before-first-import HubTool constructor). `preparation-hash-red` failed both additional result-hash controls. Foreign allocation/lookup/hash callbacks ran, initial attempts performed native reads, or issued results were adopted instead of refused. Both batches retired their owned process trees normally, released identities/pumps, removed private profiles, and recorded zero diagnostic overflow/races. These are meaningful failing controls, not passing acceptance; final correction evidence follows after implementation.

Constructor correction complete for this scope: final 92 targeted controls pass (15 new + 77 original); all normal retirement/source receipts and scoped Ruff/format/diff checks pass. Independent review found no actionable issue. TASK-34563.2 retains its wider integration/performance qualification status. Full evidence and limitations are recorded in Docs/Development/2026-10-06-console-shared-preparation-verification.md.
