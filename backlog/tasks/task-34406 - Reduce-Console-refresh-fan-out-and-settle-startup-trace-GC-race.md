---
id: TASK-34406
title: Reduce Console refresh fan-out and settle startup trace GC race
status: In Progress
created_date: 2026-10-04 19:53
assignee:
- '@codex'
updated_date: 2026-10-04 22:36
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair measured root causes from TASK-34402 under the user-authorized combined performance fix pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Live Console refresh reduces redundant synchronous storage reads while preserving current run and workspace state
- [x] #2 The startup GC admitted-revision race is confirmed or ruled out deterministically and any confirmed race is fixed
- [ ] #3 Targeted capture retry maintenance and refresh freshness regressions pass
- [x] #4 A resize during partial Console mount defers safely and applies the same width band once required rail widgets exist
- [x] #5 Credential, cost and rail presentation refreshes do not perform synchronous provider-readiness configuration IO on cold or expired owners, while explicit actions and sends revalidate the current configuration
- [x] #6 Readiness worker results identify the actual checked config source across retarget-back races; disposable or expired presentation data never replaces session defaults, and fresh eligible convergence still works.
- [x] #7 Historical Agent rail and fleet presentation performs no cold synchronous database admission, shares one finite checked read, and rejects stale profile conversation run or database results while live actions remain fresh.
- [x] #8 Opening Conversation settings with cold or changed context waits for one exact-owner finite read; completion controls are usable, and changed owners or cancelled opens do not publish stale modal state.
- [ ] #9 The test App factory physically retires its exact owned file-backed constructor handles and advisory instance-lock handle before sandbox deletion, including retained and overwritten app fields; borrowed/startup owners remain live, and active or foreign ownership preserves the sandbox and fails visibly. The original complete cohort/global maintenance drain remains unchanged.
- [ ] #10 Actual new and coalesced callers publish current tab membership before cold readiness waits; all tab publications serialize current-owner reads and remain admitted through actual completion. Maintenance waits for queued and publishing tabs, with unchanged entry refusal, cancellation, source guards and original UI deadlines.
- [ ] #11 Original surviving-child journeys retain one exact runtime/controller fixture bridge; captured fixture-owned request-worker database handles physically retire before sandbox deletion. Foreign live ownership refuses visibly and preserves the sandbox, while original actions, approvals, waits and source guards remain intact.
- [x] #12 Stock file-backed Workspace scope reads retire new worker handles and exact leases on the same worker before returning or raising; cancellation retains them through real callback completion. Borrowed handles and transactions, memory and custom owners preserve their existing lifetimes, and subsequent reads reopen normally.
- [ ] #13 Application shutdown reaches cooperative thread cleanup when the Windows subprocess registry is absent; existing POSIX process snapshot, timeout and per-process error behavior remain unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-10-04-console-performance-fixes.md. Reproduce RED, implement minimal fixes, verify targeted native and authority regressions. ADR check: UI batching N/A; native evidence and trace lifetime amend ADR-126/097 as needed.
Reproduce partial-grid resize before rail children exist; defer without consuming the width band, then verify same-band retry and mounted navigation/teardown tests. This lifecycle fix preserves existing geometry and tokens; no new ADR required.
Reproduce UI refresh contention with the real rebuild and file locks held by a worker; add presentation-only try-entry that releases partial acquisitions and coalesces one fresh replay. Default and action operations retain waiting and checked native lifetime. ADR required: yes; amend backlog/decisions/126-complete-local-backup-and-recovery.md before changing the cross-module operation contract.
Use third native probe stacks to reproduce readiness configuration IO on UI presentation paths. Capture one finite current configuration projection off the UI loop under exact config path/generation, database, session, workspace and settings fences; defer cold owner presentation and retain same-owner already rendered state while pending. Keep live action/controller configuration reads unchanged. Amend existing ADR-126 before the presentation data interface and verify profile/session/settings changes across await plus credential expiry and evidence revision.
Reproduce the reviewer-confirmed retarget-back race with an actual checked config operation and the durable default convergence defect through the real Console session/store. Tag each finite mapping with the checked selected config path and generation after entry and after loading/deepcopy; reject either edge differing from the requested source and recheck full UI ownership. Presentation session reads reuse established settings and cannot converge from disposable data. Apply eligible convergence once from the freshly verified worker mapping outside presentation scope, then recapture the resulting settings revision. Unscoped action convergence remains live. ADR required: yes; amend ADR-126 for actual-source result tagging and explicit convergence handoff before production.
Use fifth native seam evidence to reproduce cold Agent rail/fleet history IO with a real file-backed AgentRunsDB and counted repository operations. Share one screen-owned finite historical derivation between overview and fleet rows, capturing the exact bridge/database and owner/run/config identity before await; preserve all live/fleet precedence, direct bridge history/action behavior and borrowed connection retirement. Reuse current owned DB callback scope, with a captured database argument preventing redirect during await, and refresh only the Agent section when published content changes. Extend disposable readiness to the pure mode-bar presentation path identified in the fifth sampler; controller core-state selection remains live. ADR required: yes; amend ADR-126 for the finite historical presentation interface before production. Verify cold and repeated main-thread operation counts, stale owner/receiver/live-run rejection, cancellation retry and direct-read positive controls; retain original native budgets and checked config lifetime.
Review reproduced a cold settings modal retaining the disposable Loading context state permanently. Before modal construction, await a fresh finite context read under exact controller/store/session/context and settings-origin fences; reject changed ownership across subsequent awaits. Keep ordinary presentation scheduling and live context actions unchanged. Verify cold and changed context controls, stale owner refusal, cancellation and existing modal transfer behavior. ADR required: no; action initialization fixes a routine lifecycle bug within the existing ADR-126 finite read boundary.
Reproduce new/error/cancelled stock Workspace scope-read leaks with six isolated real SQLite ownership controls. Wrap only LocalWorkspaceRegistryService.get_workspace_scope's existing connection interval in operation_owned_connection(self.db), preserving scope validation, query and errors. Verify new handles and leases retire on the producer thread, borrowed transactions and memory/custom behavior remain, then rerun the original Library delete/undo and constructor-factory retirement checks. ADR required: no; routine repair uses the existing finite-operation ownership API under ADR-126/097/179, without changing storage authority or schema.
Reproduce Windows CPython subprocess._active=None skipping cooperative cleanup with six finite source-extracted original-loop controls, including the actual native registry. Normalize only the absent collection before the original snapshot; retain stop targets, process timeout/kill behavior and shutdown budgets. ADR required: no; routine platform representation bug uses the existing lifecycle contract. Verify native RED/GREEN and actual shutdown diagnostics.
Reproduce receiver retarget during the original same-worker borrowed-cache inspection with two real WorkspaceDB objects on the same file. Capture the database once at read entry and use it for both finite ownership and connection acquisition; verify exact borrowed transaction remains and no unowned B handle/lease opens. ADR required: no; routine owner consistency repair within existing ADR-126 finite operation contract.
Retain the genuine source-qualified integrated two-handle Fleet fixture leak as baseline. Install five isolated native controls for the exact captured fresh constructor Runtime and contained DB helper; missing helper is excluded from RED. Dispose that Runtime on its original live fixture loop after harness shutdown using the unchanged API grace, then fence and physically retire only settled exact declared creator caches. Scope only the existing prepared chat-create request callback to retire new worker handles, preserving borrowed transactions. Convert only the fixture/three callers to async context management; retain the directory on refusal. No precreation, changed guard, assertion, ordering or deadline. ADR required: no; reuse existing Runtime.dispose and operation_owned_connection contracts under ADR-097/126/179.

Native6 proves the three local Resend fixtures press while a changed-settings readiness projection is still pending; its real current-owner publication clears the block about 0.22 seconds later. Share the existing composer-selector two-second setup allowance across unchanged provider selection and actual checked readiness/control-bar completion in only those three fixtures. Preserve the synchronous selection helper, all provider actions, response assertions and waits, and the original 0.6-second poll check. Verify eleven source-compiled ownership controls and the three real private-profile journeys; pending product blockers must still fail. ADR required: no. ADR path: N/A (existing ADR-126 display publication contract). Reason: routine fixture synchronization; no product authority or runtime boundary changes.

Local resend fixture refinement before implementation: the first Native barrier over-waits unrelated whole-sync work; current projection refresh retires and issues its checked proof before scheduling that publication. Require only the exact current projection refresh/pending state to settle and the actual original checked control-bar return True. Positively hold unrelated same-screen whole-sync work while current readiness succeeds; preserve the same original two-second shared allowance and every response/action/assertion. ADR required: no; routine fixture synchronization within the existing ADR-126 contract.

Local selected-readiness poll refinement: actual Textual Pilot.pause first waits on every screen descendant regardless of requested sleep, reintroducing unrelated completion and allowing the two-second fixture deadline to overrun. A pure test compiling the pinned real Pilot.pause body reproduces that wait. Poll current checked readiness with bounded asyncio.sleep instead; preserve the selector, two-second deadline, original three provider/action journeys and every response/0.6-second assertion. Verify that meaningful RED/GREEN control and original native journeys. ADR required: no; routine fixture synchronization and driver timing correction.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Startup collection between admitted preparation and boundary reservation reproduced trace_revision_unavailable. Live canonical revision metadata and FK ancestry now survive, while separate graph-only payload roots allow detached owner bindings and bytes to purge even when a second owner retains the same frozen policy. ADR-097 amended; 24 GC tests pass and TASK-33621.47 is Done. Presentation context reads now share an exact owner/revision memo across refresh, credential and spend callers; cold and expired values schedule finite owned worker reads, and explicit controller action authority still reads live. Real AgentRunsDB counts use bounded per-row-set owned reads with identity fences and cancellation retry. Exact completed receipt acknowledgements dedupe under the existing attention operation lock and clear on detach. Shared-policy privacy and removed-session error regressions added. Partial-mount resize reproduced five failures; required child queries now defer without consuming a width band, all five regressions pass, and all five real mounted teardown/navigation tests pass. Existing geometry and token values retained. Native evidence identified UI blocking on worker-held config rebuild/file locks; four deterministic tests reproduced 2.09-2.19s stalls and now pass with less than 100ms projection return, responsive heartbeat, one fresh replay, released partial lock/acquisition and unchanged config state. Presentation-only try-entry preserves continuous checked native body and default explicit-action waiting; ADR-126 amended. Expanded config lifetime/maintenance and frozen refresh verification are running. Final combined native performance measurement remains required before AC1/3 and task completion; no timer suppression or ceiling changes.
The third native probe remained RED and exposed synchronous provider readiness configuration loads in credential, cost and rail presentation. The shared presentation mapping now reads current configuration in one finite worker under the existing continuously checked config lifetime, with exact config path/generation, app/database, session/workspace and settings revision publication fences. Cold owners defer; expired same owners retain their last rendered mapping while pending. Live default getters, actions and sends retain current authority. Ten readiness and credential expiry/background completion tests pass (56.16s), including five post-await owner swaps, cancellation retirement, actual ChatScreen decorators and a live getter control. After this change, the real config lifetime/lock/action subset passes all 17 selected tests (92.15s). The prior full config lifetime and maintenance bundle passed 34 tests. The obsolete full-log test now awaits the current owned probe and retains its original primary/oldest/newest target assertions; that exact node passes (20.97s). The unchanged badge query-count node passes in isolation. Ruff scoped helper/new tests and diff whitespace checks pass; the next isolated combined native probe must still meet the original budgets before AC1/3 and completion.
Reviewer follow-up reproduced an authoritative session defect: the real ConsoleSessionController and ConsoleChatStore changed an established pristine model to the expired disposable presentation model. Presentation scopes now reuse established settings and skip convergence; a worker result carries generation and actual checked raw selected path at both callback edges, then one fresh verified handoff outside that scope applies eligible defaults and recaptures its own resulting settings revision. Live unscoped action convergence is preserved. The source-provenance RED varies the cheap UI identity A to B and back while a real checked operation reads installed B; the prior code published B under requested A. A literal environment retarget to another file is separately refused before the reader body by the unchanged installed participant with raw_source_selection_changed, so this is no claim of a reproduced native guard bypass. All 18 focused readiness tests pass (5.33s), covering both source edges, active-operation provenance, real convergence and expired reuse, session/workspace swaps, cancellation, live action control and pristine provenance/default generation gates. ADR-126 amended before production; the new check supplies data provenance only and no authority. Final combined native performance gates remain pending.
Fifth native evidence still failed the original performance gates, but isolated cold Agent history presentation was confirmed to admit two synchronous list_runs calls. Exact file-backed ConsoleAgentBridge presentation now shares one finite owned historical derivation between the overview and fleet, captures the database receiver for both queries, rejects changed bridge/database/profile/conversation/run owners after await, retires cancelled callbacks before retry, and schedules only the existing coalesced Agent-section refresh. Live fleet/overview precedence and direct bridge historical/action behavior remain unchanged; memory and custom bridge compatibility paths are preserved. The real SQLite counter regression measures two main-thread admissions before versus zero after: both SQL queries share one worker counted callback, while two existing checked connection-close intervals remain, so this is not an all-thread admission reduction claim. The pure mode bar also reuses the established disposable readiness projection; controller provider/runtime selection remains live. ADR-126 was amended before this interface. The original regression was RED on the cold main-thread counter, then the final formatted bundle passed 36 tests (31.46s): 30 refresh/readiness tests, two mounted stub compatibility/live-precedence controls, and four direct bridge historical controls. Two real mounted persisted Agent copy/unchanged-paint controls passed in the prior run (14.60s and 12.44s). The two stub nodes initially failed before app creation due to their missing bootstrap_profile fixture; adding that exact profile owner marker retains all original assertions and native guards. Scoped Ruff and diff whitespace checks pass. Screen class size is 18 lines smaller than current HEAD with the same 766 methods; no ceilings changed. Production/tests frozen with hashes for the sixth native measurement. AC1/3 and task completion still require the original combined native gates; no timer suppression or performance budget increases.
Independent review reproduced cold/changed Conversation settings permanently retaining busy Loading context and disabling Apply, Save and Make default. The action now awaits the existing exact-owner finite reader, rejects changed session/payload/controller/store or durable settings origin after awaits, and then constructs a loaded immutable modal. Ordinary presentation and live context mutation authority remain unchanged. Two original cold/changed cases were RED on busy=True. Final formatted targeted bundle passes 21 cases (21.80s), including the actual mounted enabled Apply/Save/Make default controls, changed owners during model await, retained worker/native custody on cancelled opening, and existing modal mount/transfer and presentation controls. Initial cancellation fixtures gated before handle acquisition and were corrected to assert the actual acquired handle; no production cancellation contract changed. No pending-task diagnostic remains in final run. Routine lifecycle repair follows existing ADR-126; no new ADR required.
Final modal follow-up: chat closure during both model lookup and the actual gateway thinking-policy resolution originally raised KeyError before publication fences. The shared exact-owner helper now runs immediately after awaits and before modal dereferences; KeyError from thinking resolution is converted to False only if the captured origin has gone. Final source targeted bundle is 23 passed, 432 deselected in 21.62s (normal exit 0), including all three mounted completion buttons and existing modal transfer/mount controls. No pending worker/task diagnostics in this final receipt. Final chat_screen SHA256 21abec1ebdbf4a6c32344f880183597fdc6d90521ea1556307018bf128b8988c; new modal test 5c863ad9cd4cde9b92c498616ddf40295d1970cb378018f3473784f635be2c0a. Source frozen before final native measurement.
Source-qualified Library cleanup attributed the remaining Workspace worker lease to LocalWorkspaceRegistryService.get_workspace_scope through the original executor callback. Three native new/error/cancel controls failed on the actual live SQLite object and exact lease; borrowed, memory and custom controls passed. The existing operation_owned_connection boundary now covers only that synchronous producer interval. All six native controls pass (17.35s), including borrowed transaction preservation and retained running-worker custody after waiter cancellation. The unchanged original Library delete/undo journey passes (37.73s); its bounded original factory witness observes all six declared owners, successful physical cleanup and no refusal, with current sources, no invalid evidence and retired local hooks. This proves that producer repair, not overall startup/Send budgets or the original complete-cohort drain. ADR required: no; existing finite ownership API.
Native Windows source-extracted shutdown controls reproduced two absent-registry failures and four unchanged POSIX behavior passes. The one-expression absent-collection normalization now passes all six controls (5.94s driver), with current before/after sources. Actual Windows registry None and cooperative stop are covered; no shutdown budget or target changes. The integrated shutdown and whole pause gates remain pending.
The receiver-retarget native regression reproduced an ownership/query mismatch: borrowed A retained its transaction while query B opened an unowned live native handle and lease. One captured database local now supplies both the ownership interval and query. All seven native lifetime and retarget controls pass (24.97s driver), with current unchanged source hashes, exact physical retirement and zero network/real-profile guard effects. AC12 is complete; broad pause and fixture cohort gates remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->


## Exact non-chat fixture Runtime ownership follow-up

Acceptance addition proposed for task-34406 before managed implementation:

- Every fresh non-chat-create app yielded by `_pending_close_app` retains its exact constructor-owned stock Runtime and original creator thread/running loop before the harness mounts. After the original harness exits, the existing Runtime disposal completes and its actual watcher/read task positively retires. Built/pre-existing, foreign, replaced, custom Runtime or wrong thread/loop refuses ownership; the fixture never adopts/closes a DB or sweeps other owners.
- Cancellation of the fixture awaiter, including repeated cancellation, does not cancel or abandon its one retained original Runtime disposal task. Actual retirement completes on the original loop before cancellation propagates. Original pending-round setup, requests, effects, guards, arming assertions, order and deadlines remain unchanged.

ADR required: no
ADR path: N/A; existing Runtime.dispose contract and ADR-097/126/179 apply.
Reason: Exact test-fixture Runtime lifetime cleanup reuses the existing production API; no production storage/runtime authority or lifetime API changes.

Plan:

1. Install the retained regression alone and run its genuine original non-chat fixture route before cleanup repair. It must establish the real stock Runtime watcher precondition, fail the retirement assertion after the unchanged context exit, and only then use exact test-owned Runtime cleanup. Missing-helper/import/setup failures are not product evidence.
2. Install one narrowly scoped Runtime owner helper plus isolated real-stock Runtime controls. Leave PreparedCloseOwnedResources bytes, fields, database declarations and its source-witness contract unchanged. Check stock fresh ownership, wrong/built/foreign/changed Runtime, wrong thread/loop, actual watcher/read retirement and repeated caller cancellation under a held original policy read.
3. In only the existing `kind != chat_create` branch, construct this owner before yield and dispose after the original host exits. Preserve a primary journey error with fixed static cleanup reason/type notes. No DB adoption, close, precreation, actor changes, modified original assert/wait/deadline or global cleanup.
4. Re-run retained branch and focused controls; then the unchanged original fleet journey with current source-qualified lifetime/origin witness. Isolated helper success does not establish pending-round success, Characters worker settlement or global drain.

Source audit: `prepared-fleet-fixture-ownership-draft/non-chat-runtime-source-audit.json` is source-only. Native Characters diagnostic7 shows worker operations already present at disposal START; that separate original-owner ancestry gap is not attributed or repaired here.
Retained genuine native regression before repair: original-nonchat-runtime-retirement-native-red-1 fails on the actual original non-chat Runtime still active after fixture exit (14.61s), with sources unchanged and test-owned cleanup only afterward.

Non-chat Runtime lifetime implementation evidence: retained original branch RED14.61s; all9 exact-stock native controls GREEN34.56s, source unchanged, original default disposal grace and repeated waiter cancellation custody retained. No new DB adoption/close; original integrated Fleet/complete-prefix acceptance pending.


## Shared storage coordinator performance follow-up

- [ ] Independent issued-state observation and actual original lease retirement finish while another actor performs a fresh native scope/path proof; original authority, pause, owner, cancellation, hold and source-change refusals remain intact.
- [ ] Original Send/startup/UI and platform limits remain unchanged and pass before completion.

Implementation plan (before test and production changes):
# Fresh storage proof outside the coordinator: implementation plan

> **For agentic workers:** Use superpowers:executing-plans for the root-owned
> serial Native TDD sequence. This draft author performs Evidence-only work.

**Goal:** Independent issued-state observation and actual lease retirement can
complete while another original actor performs a fresh native storage proof.

**Architecture:** Reuse the existing repository proof-before-lock pattern.
Keep every native guard fresh, count acquisitions/leases at their existing
boundaries, and put only exact metadata validation/publication under the shared
coordinator. A live-hold fallback is a per-call captured dependency, never a
permission cache or an excuse to move shared table reads outside their lock.

**Tech Stack:** Actual CPython 3.12, real Windows native admission/RLock/SQLite,
local sys.monitoring START/RETURN/LINE controls; existing private-child runner.

**Spec:** TASK-34403 current accepted native custody and performance criteria;
existing Docs/superpowers/plans/2026-10-04-console-performance-fixes.md.

ADR required: yes, existing ADR126 amendment before production.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: scope-proof publication crosses the shared storage ownership boundary.

## Global constraints

- No guard/callback replacement, weaker privacy/path/source check, Native cache,
  widened namespace, maintenance capability, timeout or performance-cap change.
- Preserve initial/full/final startup permission, all scope membership checks,
  native hold, accepted pause semantics and count-before-final observation.
- Preserve custom/unqualified native outcomes and exact startup readmission.
- No App or Native launch by this draft author; root owns serial qualification.
- Actual same-source evidence tables and epoch are read/written under _lock.

## Review focus

- A last independent token may retire while proof runs: unrelated close must
  finish without transferring the retired hold's continuation authority.
- A pause/cancellation may start during proof: new ordinary acquisition refuses;
  already counted exact operation work retains its established pause contract.
- Source/selector/root/path/actor changes during proof must not publish authority
  from an earlier selection, including a changed installed operation or lease.
- Another acquisition may install/retire/change a hold during proof: reread exact
  hold/names/readiness/error and the evidence identity/epoch under the coordinator.
- Native/body/close failures keep existing cleanup/error precedence and uncertain
  resource ownership; no automatic success, empty witness or unknown retirement.

## Source findings, not runtime attribution

Current storage_admission._acquire_storage calls _scope under _lock in its first
scope publication and final revalidation blocks. _scope reads _records,
_registry, _binding/fingerprint/root/path proofs, including a native registry
shared lock. Locked check() in that caller and _reuse_evidence may call
_Operation.check(path), so its internally separated native stat/resolve is still
inside the caller's outer reentrant coordinator. _repository_operation and
participants._core_getter already perform full native proof before their short
pure final coordinator fences.

Fleet stage 6 establishes a 10.141-second first get_run admission window, with
the raw connector only .062 seconds. Its selected events do not establish which
actor owns the shared coordinator or separate lock wait from native work. No
exclusive CPU, whole-Send attribution or optimization saving is claimed here.

### Task 1: Qualify actual causal controls before any production edit

**Create:** test_storage_coordinator_native_io_draft.py in EvidenceRoot.
**Test:** three private-child routes scope_before_count, scope_after_count and
operation_check. Root may run the Evidence file directly without managed copy.

- [ ] Run genuine Windows private-child source-current RED. The original opener
  must be reached under exact acquisition/record-reader or operation ancestry.
  The control suspends its unchanged body, never replaces it. Exact actor,
  installed operation/lease, root/selector/path, original code/globals/defaults,
  module origin/bytes, real RLock and original guard identities must qualify.
- [ ] While the native body is held, a second actual Thread performs a
  nonblocking coordinator acquisition, then invokes the original close on a
  separately admitted live token. Its actual close LINE reaches the coordinator.
  Require lock entry and original close completion before release. Native body
  release/join, seeded physical SQLite close and final zero census precede the
  causal assertion, so a genuine RED cannot abandon a native owner.
- [ ] Classify boundary/setup/source/cleanup failures separately from product
  RED. The scalar receipt is written before the expected causal assertions.
  Global monitoring is zero, events bounded, and local tool physically retired.

### Task 2: Narrow original proof/publication split after qualified RED

**Modify only after root authorization:** storage_admission.py checked admission
and reuse seams; add metadata helper only if it makes the existing pure fences
explicit. Do not change raw source, repository, guard or native helper APIs.

- [ ] For every currently locked attempt.check(path), obtain the unchanged full
  native operation proof outside _lock. Inside _lock recheck exact pending
  acquisition membership, PID/Thread/Task, cancel, original operation identity,
  installed owner/lease/path/hold and the accepted pause semantics. Preserve all
  selected/related-path guards; do not collapse independent proofs by memo.
- [ ] Split _scope's shared metadata dependencies from its original fresh I/O.
  Capture its exact current hold/names and startup pause-owned source/roots/
  authority identity under _lock; keep _records/_registry/_binding/fingerprint/
  effective-roots/contains proofs fresh outside. All mutable shared table reads
  stay synchronized. Native callbacks and refusal behavior remain installed.
- [ ] Record whether the unchanged live-hold continuation branch actually
  supplied mapping authority. Only that branch depends on a retained live hold:
  recheck the same current hold/names/live state before accepting its result.
  A fresh valid saved binding does not fail merely because unrelated count
  changed; a retired fallback hold cannot lend its previous mapping. Startup
  readmission retains its exact issued pause and authority checks.
- [ ] Reenter _lock after each proof. Recheck actual actor/source/current
  selection, hold/names/readiness/error and captured dependencies before creating
  or reusing a hold and publishing a token. Keep token/pending counting before
  the final fresh proof, and keep final fresh startup permission plus scope
  validation outside the coordinator with a final pure publication fence.
- [ ] Preserve _reuse_evidence's synchronized hold/evidence/path-table capture,
  counted token and per-call fresh native observation. Move its native attempt
  checks outside with exact operation/hold/evidence/epoch postproof fences; no
  evidence or scope decision crosses a later call/await/actor.

### Task 3: Refusal and compatibility evidence before acceptance

- [ ] Three causal controls GREEN with original bodies/source stable and zero
  final native counters. A last-token close cannot be replaced by a fake lease.
- [ ] Held completed-proof controls: actual pause/cancel; removed issued
  operation/lease/participant; changed owner/path; changed root/selector/source;
  retired/changed hold and changed evidence epoch/identity. No custody/result
  publication from obsolete proof. Valid independent count change still works.
- [ ] Target original repository coordinator, raw-source pause/provenance,
  storage evidence oracle, startup readmission and native close-retention cases.
  Run corresponding original POSIX checks without simulating Windows.
- [ ] Static/source review then one serialized unchanged original whole probe.
  No whole budget pass or speed claim follows from a held-control test alone.

## Alternatives rejected

Running I/O on a worker while retaining _lock still blocks retirement/main actors.
Private RLock release/restore obscures scope ownership and publication races.
Moving _scope as-is races _holds and pause/startup metadata. Caching resolved
permissions, removing checks or using faster unqualified path APIs weakens
freshness and source/privacy boundaries. A new per-profile mutex, precreated
store, broadened generic worker admission or increased budget is outside scope.

The three drafts have not run Native. Root must review and obtain actual RED
before deciding the exact amendment and production edit. No completion claim.

## Storage proof follow-up verification (in progress)
Storage coordinator native proof repair: the three unchanged-body Windows controls reproduce the shared RLock blocking an independent original lease close before the edit (storage-coordinator-native-red-1). The raw registry read and nested operation path checks execute inside the outer coordinator; all three actual native reads return and physical SQL/native ownership retires before each causal RED assertion. The repair captures synchronized dependencies, performs original fresh I/O outside the coordinator and repeats exact actor, pending, installed operation, pause, selection, hold, continuation and evidence-epoch checks before publication. ADR-126 records the amendment before implementation.

Final formatted-source storage-coordinator-formatted-native-green-2 passes all3 controls (16.75s driver, source unchanged). storage-scope-publication-native-1 passes6 actual held-return races: cancellation, pause, selection change, retired continuation, a valid saved binding after unrelated retirement and revoked installed operation. The original compatibility selection passes53, skips41 platform cases and fails1 Windows related-symlink fixture. The bounded unchanged-body diagnostic original-related-path-refusal-native-2 qualifies the original fixture's Path.symlink_to exception errno22/winerror1314 before its containment proof; it does not establish a production refusal regression. POSIX and elevated Windows matrix verification remain required. No guard, callback, persisted enrollment, check freshness or budget was relaxed.

Integrated Fleet native8 preserves current source and reaches the original later fleet-close scenario. Its sole child failure is the unchanged cancelled.is_set() and round_task.done() settlement assertion at test_console_session_tab_close.py:1035; no constructor-resource cleanup failure is reported. The diagnostic selected no applicable fixture owner, so its false source/coverage flags and empty rows provide no origin attribution. This is not a Fleet pass or a timing pass.

Boot inventory follow-up plan: exact309b Perf Guard reports one unreviewed pair (_sync_native_console_chat_ui, console-sync). The existing coalesced replay after readiness/selection drift invokes the same whole Console synchronization needed for current visible state. Record that feature-owned conditional pair in the existing membership inventory, preserve mandatory-start set, scheduling policy and all module/concurrency/time budgets, then verify original census and policy cross-checks. No new ADR: existing ADR-126 same-owner readiness publication/replay.

Character rail publication follow-up plan: the original whole probe records character_context._publish -> wiring._sync_character_context_presentation -> _current_console_rail_state -> inspector/readiness synchronous load_settings calls outside a checked display scope. Verify an actual warmed issued projection does not open main-thread native/config scopes during that original callback; obtain RED before wiring the render-only rail calculation through the existing _run_console_config_sync API. Keep widget/Character state publication, cold/expired retry, original body errors, live action reads and same-owner proof fences. ADR required: no; existing ADR-126 disposable presentation boundary applies.

## Historical cancellation follow-up acceptance and plan

- [ ] A historical presentation read keeps its exact pending state and callback ownership across repeated cancellation until the original callback and owned native handle retire. It cannot publish or rearm early; borrowed handles retain their original worker lifetime.
- [ ] Single, double, triple cancellation, borrowed double cancellation and current successful publication have actual original-source native evidence.

ADR required: no new ADR. Existing ADR-126 finite callback and same-owner publication boundaries apply. Before the production edit, qualify the unchanged original callback held after its first real SQL query; then repeatedly shield the same captured Task through further awaiting-worker cancellations, consume its terminal outcome and propagate original cancellation. Preserve current/stale source fences, admission, physical handle policy and all budgets. Run existing refresh batching controls and original whole evidence after the narrow repair.

## Evals fixture ownership follow-up acceptance and plan

- [ ] The private factory records and physically retires its exact constructor EvalsDB even after app fields are replaced; borrowed or foreign-path owners are excluded.
- [ ] Unchanged original normal and queued MCP wiring tests leave no constructor Evals connection or ordinary resource lease behind.

ADR required: no new ADR; this extends the existing test-owned constructor inventory and ADR-126 original-owner retirement proof. First retain actual constructor EvalsDB and native handle through original directory teardown to reproduce RED. Then redirect its constructor path into the existing private factory directory and include only its exact type/path owner in the existing inventory. Preserve native close/admission/census checks, foreign-worker refusal and borrowed owners. Verify retained/replaced-field controls and existing constructor retirement regressions before original shared-cohort checks.

## Recovery presentation follow-up acceptance and plan

- [ ] The three stock recovery display builders read their established current controller state without refreshing executable provider/agent-runtime selection on each repaint. Cold/custom/retired-owner routes and explicit card actions preserve live ensure/core behavior.
- [ ] Actual mounted Console regression counts original ensure/core/settings entries and retains equal recovery state. Existing live binding, pre-dispatch and continuation recovery controls remain passing.

ADR required: no new ADR; existing ADR-126 display versus live action and ADR-220 live recovery binding boundaries apply. Obtain actual original mounted callback RED, then introduce one read-only established-controller accessor qualified by the exact runtime/app/view/generation/controller/store pair. Wire only the display getters to that accessor, preserving original callback invocation and action ABI. No cached mapping, permission or dispatch authority is introduced; actions continue their fresh original ensure path. Verify owner/custom/cold fallback and existing recovery actions before original whole performance evidence.

## Remaining native work-count qualification plan

- [ ] Fresh Windows binding observations cover every actual enrolled root and all ancestors with original owner/mode/sticky, native identity and close-retention policy; no observation crosses a binding boundary.
- [ ] Changed Character display preserves initial callback retirement and original REFRESHING publication; only its inner refresh pairs/recent groups share a finite exact-DB callback, with all original midpoint and post-retirement ownership checks.
- [ ] Actual original-body work-count RED precedes either optimization, with native physical handle retirement and source hashes. Custom, borrowed, drift and refusal controls retain original paths.

ADR required: existing ADR-126 amendment before production Character batching; existing fresh metadata observation policy applies to a Windows-only binding leaf. No new service, owner, authority or persistent cache is introduced. Read the retained proposals and run their actual original-source controls first; setup-only errors do not qualify as product RED. Original whole-Send/startup/helper budgets remain mandatory after the leaves, and a leaf saving cannot establish overall acceptance.


## Original notification delivery prerequisite

- [ ] The original live-screen callback journey waits for actual asynchronous notification delivery before inspecting its existing message and privacy assertions. Delivery timeout and all original keep-alive, screen, focus and quit checks remain intact.

ADR required: no; routine test-driver synchronization. The source-qualified unchanged original native callback fails because inspection precedes original Textual delivery (inspection age0.342s, delivery age0.499s; timeout12s, no expiry, all original callback pairs complete/current, monitoring retired). Before changing the test, retain that RED receipt. Add one bounded existing _poll delivery prerequisite, preserve the existing0.3s screen-settlement wait and original final assertions; verify both content and modal callback cases with actual source-current native execution.


## Startup cohort diagnostic inventory follow-up

- [ ] The original diagnostic launcher accepts the exact authorized targeted workflow prefix including the four added storage controls, while missing, reordered or unknown entries still refuse before native launch.

ADR required: no; existing diagnostic membership inventory. Both exact8c CI launchers refuse before native launch because their fixed expected PREFIX lacks the four added storage modules. Preserve that preflight RED; add only those four exact entries in their actual workflow order. Keep the original parser, snapshot checks, node order, source fences and every deadline unchanged. Five pure parser/actual snapshot-guard controls verify acceptance and omission/order/unknown refusal before implementation. Original actual cohort execution remains required on next pushed source.


## Hook-key Workspace callback ownership follow-up

- [ ] Actual background hook-key consent and policy reads physically retire newly opened Workspace handles on their producing worker; the original chat-store scope, fresh authority reads, custom/memory behavior and borrowed handle ownership remain intact.
- [ ] Registry/database retarget, body failure and repeated cancellation refuse stale publication and cannot release ownership before physical callback retirement; original retry teardown retains exact-owner native evidence.

ADR required: no new ADR; routine repair applies ADR-126's existing finite supported callback ownership contract. Original retry diagnostic3 positively retains seven exact constructor owners, two real Workspace worker handles/leases and their original source-qualified hook-key consent/get_workspace call chains. The retry body passes but exact factory close refuses with those handles and a distinct availability callback active. First qualify direct native callback lifetime RED controls after real callback retirement. Extend only the captured hook-key Workspace producer scope; where needed, original service readers capture their exact database and retain supported owned-connection cleanup so retarget cannot redirect a callback outside its captured owner. Preserve original fresh queries, actions, policy errors, custom ABI, borrowed transactions and all original budgets. Verify physical retirement and owner drift before rechecking the original retry. Treat the separate cancelled availability callback lifetime as its own hypothesis until positively reproduced.


Windows binding implementation refinement before production: ADR-126 now records the exact definition-time Windows tree-reader record and custom scalar fallback. Original immediate-parent missing-leaf policy, trusted owner delegate and drive-root refusal remain; the preceding wrong-policy candidate is withdrawn. Install and verify baseline parent/source/custom Native controls first, then apply only _binding plus the defining metadata record. Verify actual selected body drift and protected-HANDLE uncertain-close custody distinctly from observer-only source refusal; no broader batch or permission reuse.

Hook grouping refinement before production: the qualified stock outer ownership scope must retain one actual database.connection interval so both fresh original Workspace readers share one native handle; a lazy owner context alone opens nothing. Native controls assert exact one-handle normal grouping, borrowed-A/new-B retarget retirement and separate direct-reader handles for preinstalled Runtime/controller/registry alias replacements. Defining class identities join the original method records; custom/injected compositions retain the preceding route. Existing ADR-126 applies. The nine original callback baseline outcomes qualify five lifetime REDs and four ownership positives before installation; new inherited-alias setup controls precede production.

Character entry refinement before production: capture exact inner batch/scope receiver-functions and verify the defining metadata again after original queue/native admission, before either captured mutable body runs. Native controls hold actual queued/admission/inner entry boundaries, replace only the two new unguarded reader bodies, require zero changed-body entries after qualification and positively retire original handles/leases before assertions. Preserve original outer capture/_begin/midpoint/generation/error/cancellation boundaries and direct/custom ABI. The frozen6c04 draft is a temporary causal checkpoint only; final production must include the reviewed entry repair. Existing ADR-126 applies.

Workspace availability cancellation follow-up (before production)

- [ ] Cancelled Workspace availability refresh retains its exact native callback and in-flight producer through physical retirement under repeated cancellation, publishes no cancelled result and cannot rearm a second producer while the first callback is alive. Fresh named/Default projections, post-retirement retry and same-worker borrowed transaction custody remain intact.

ADR required: no new ADR; existing ADR-126 finite asynchronous callback lifetime applies. Qualify the original seven Native cancellation/current-success controls at actual SQL and operation/lease boundaries. Only after genuine lifetime RED, retain one task for each existing selected callback, shield/drain the same task through every cancellation before the original finally releases its in-flight claim. Preserve both original dispatch routes, generation/registry/database publication fences, guards and stage bounds. Recheck original retry and exact seven-owner teardown afterwards.

Original asynchronous test prerequisites (before changes)

- [ ] Original Character setup nodes explicitly use their existing private bootstrap profile, and cold/resume readiness waits for the exact current idle widget projection/DOM within unchanged40/.05 polling bounds. Original assertions, worker counts and actions remain.
- [ ] The original performance probe waits for actual provider settlement retirement inside its unchanged15-second captured Send phase before unchanged durable trace/link assertions; expiry/cancellation cannot close an active worker.
- [ ] Cold finite Workspace composites preserve the preceding single physical handle through all fresh queries/writes, with supported exact-owner retirement and borrowed transactions preserved. Actual native counts precede any composite grouping repair.

ADR required: no new ADR for fixture synchronization; existing ADR-126 applies to finite composite lifetime. Original Character run has16 setup-only raw-source-selection errors, one post-resume DOM failure (oldrevision alone was accepted while new projection was loading),91passing bodies and a separate unassigned factory teardown refusal. Add four precise bootstrap markers and complete-state/DOM prerequisites within originalbounds. Exact47a9 Ubuntu trace receipt reaches its terminal oracle while the third original settlement worker is still owned; original prepared/claimed/store_run complete31-34ms later and teardown drains tozero. Add only a bounded pending-work retirement prerequisite inside the original Send allowance, then freshly re-read original durable state/link assertions. Qualify held-real-settlement, expiry/cancel controls before originalwholeprobe. Cold-composite candidate Native controls cover originaladmit/status/capture/save-binding queries and positive borrowedtransactions; no globalhelper-depth, authoritycache or proxyregistry change.


Cold receipt initialization qualification (before production)

- [ ] Original cold Console composition keeps its actual loop responsive while its real receipt-schema SQL/native handle is held. Original storage/bridge/provider/readiness, live first Send and normal disposal remain intact.
- [ ] Any accepted asynchronous preparation retains its exact callback through repeated cancellation and disposal; source/runtime/profile/navigation drift refuses publication without borrowing or closing unrelated handles.

First qualify the actual original receipt initialization path with the real connection backend before making a production startup change. Three preliminary native runs fail observer prerequisites and are excluded: a cleared Runtime field, an assumed caller, and an assumed sqlite3.Connection. The third records the source-qualified original Inspector/bridge/receipt/schema ancestry; The observer records only that the exact base sqlite3.Connection type does not match; original connector source uses an AdmittedConnection native subclass carrying its StorageLease. The exact-type mismatch alone does not prove a process facade. The corrected control must retain the actual admitted native connection/lease, keep global monitoring zero and positively retire that original resource before asserting shared-loop responsiveness. No precreated store, guard bypass or original timing-limit increase is permitted. Production remains pending genuine causal RED and amendments to existing ADR-085/ADR-126 before implementation. Prefer existing async initial-screen custody to prepare only receipt storage, preserving synchronous/custom/headless Runtime ABI and UI-bound bridge construction.


## Cold Workspace composite repair refinement (before production)

# Four proved cold Workspace composites: proposed correction

ADR required: yes
ADR path: existing canonical ADR126; proposed append is `adr126-refinement-draft.md` beside this plan.
Reason: one optional finite source-qualified connection boundary is shared by the Workspace registry, Change Review and FileInspector.

Evidence: root's unchanged seven-node native baseline completed with four genuine cold work-count failures (exactly two physical/query handles), three genuine borrowed-transaction passes, all literal result/source/physical-close/lease/observer checks qualified, driver30.25s. This supersedes no earlier failed fixture evidence and does not establish timing savings.

1. Independently review this Evidence-only five-source patch against `source-manifest.json` exact installed raw baselines. Apply the ADR126 refinement before managed production changes.
2. Retain the unchanged seven original controls. Install the seven additional draft controls, with their original source/actor/physical-close/lease/guard checks and unchanged20-second finite callback bound.
3. Run the seven baseline controls plus negatives serially on the root-owned native slot. The write-retarget control uses only a local original save-method LINE event after its actual execute returns, while its real transaction is still active; it requires actual rollback of the persisted binding label. Metadata/body retarget controls mutate unguarded original functions and restore them in worker `finally`; no production admission/native callbacks are replaced.
4. Qualify custom instance and subclass callbacks through their original public reader route and literal overridden revision. Ordinary missing-table SQL must return UNAVAILABLE, with one physically retired handle. DB/source drift must prevent any later query and retain original cleanup. Every new metadata source body and callback stays original except the exact deliberate fault, restored before final source checks.
5. Use appropriate existing Workspace/Change Review/FileInspector tests after native focused GREEN. Broader adjacent composites remain unmodified in this patch. Do not infer whole-send acceptance from work-count controls.

Self-review: capability is outside the Workspace scope; locks and SQL/query order remain original. Both metadata and mutation-generation helpers are fenced before invocation. Save's owner/source check is inside the write transaction after execute and before commit. Static original lookup qualification precedes `vars(registry)` and dynamic DB fields. Defining getter/transaction/owned-cleanup wrappers and their known original closure bodies are recorded rather than learning arbitrary unwrapped callbacks. New successful results are returned only after the helper's retirement fence.

Qualification so far: pure actual Python3.12.10 source compilation, real stdlib contextmanager/actual original core-transaction decorator hierarchy, metadata code/defaults/keyword-default/closure drift, finite cleanup order, borrowed preservation, source refusal, no-retry entry error and body/cleanup error precedence passed. These pure controls import no project modules and create no App, SQL or native fixture. Native candidate acceptance remains pending.


# Additive controls for frozen Workspace composite candidate4958

ADR required: yes, before any repair.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: exact captured native resource retirement and source selection complete the already proposed finite ownership boundary.

Independent source review identified two blockers; no Native failure is claimed here. The frozen4958 source patch is unchanged. Root should run these controls on that exact temporarily installed candidate, after its live UAT host has retired, and retain genuine RED separately from fixture/setup failure.

Three custody routes:
- new_empty: at the actual original held-connection body RETURN, the actual cache A contains a newly opened and registered native handle. Swap only database._thread_local to empty B. Source refusal must retire new A before the product boundary.
- new_foreign: B already contains a real current-worker handle registered to another original WorkspaceDB. Source refusal must retire new A and preserve the unrelated B handle/cache/lease.
- borrowed_empty: original A already has the actual current-worker borrowed handle before arming. Source refusal must preserve that handle and its lease.

The callback records the exact repository, actual Thread, captured thread-local, actual native SQLite handle and participant/lease/path identity. The receipt records the product boundary BEFORE test cleanup. Fixture cleanup proves those exact facts again and uses only the original _core_closing API and actual native close on that precise fixture-owned A handle. It does not restore/adopt/change either thread-local/cache field. Foreign cleanup stays on the original foreign receiver/current worker. All final source/guard/physical-close/actor-lease0 checks and monitoring retirement precede the sole product assertions.

Four lookup routes cover exact original Consent and Inspector classes with either a custom delegating __getattribute__ or a data property replacing _registry after ordinary construction. Actual original callback instruction selection returns A for the added capture/owner checks and B for the original body reader. A real B database/consent/folder binding supplies literal B outcomes. The preceding custom path must execute those B callbacks and two B physical readers, with zero optional A handle; accepting the changed lookup as stock is the product failure. No production guard or native method is replaced.

Pure actual3.12.10 controls reuse the exact selection AST and prove both getattribute/property instruction routes, plus local PY_RETURN mutation of an actual threading.local field after an original returning body. Both outer/embedded Native test sources compile, Ruff is clean, global mask0 and monitoring tool retired. These pure controls do not import the project or create App/SQL/native fixtures. Seven Native candidate cases remain pending root execution.

After genuine RED, the smallest planned retirement repair should use the existing original core-closing API for only a positively identified newly created captured handle after its counted interval retires. Preserve A borrower and foreign/replacement B; never restore cache fields or adopt a mutable replacement cache. Lookup repair must decline custom class lookup/registry descriptors before added source selection and preserve original custom body callback ABI. No generic helper or permission policy changes are authorized by these drafts.



## Qualified cold receipt preparation plan (before production)

# Finite initial Console receipt preparation

ADR required: yes.
ADR paths: amend backlog/decisions/085-console-activity-receipts-and-switcher-ownership.md and backlog/decisions/126-complete-local-backup-and-recovery.md; existing ADR094/139 govern App/runtime/navigation ownership. ADR179 is the hosted provider registry and is outside this repair.
Reason: explicitly define the original receipt initializer's asynchronous startup and same-App/database publication contract. No persistent preparation service, readiness capability, storage/schema change, permission cache or new authority is introduced.

Proposal before any managed production edit:
- The original resolved initial Chat route prepares only its durable receipt store before constructing its Console screen. Other destinations and original synchronous/custom/headless Runtime APIs keep their preceding routes.
- Only exact stock Runtime/helper/reader defining metadata qualifies the optional worker route. Preinstalled custom functions/instance shadows, subclasses and memory/custom database receivers retain their original route. After selection, queued/body/default/class drift refuses before replacement invocation; there is no custom fallback inside the selected interval.
- Source qualification for publication is scoped only to the selected finite worker through a thread-local proof restored in its original finally; preinstalled class/instance/subclass wrappers may continue delegating to the original synchronous reader. The proof contains the exact Runtime and its captured source-current check, and is not a service, cache, authority or reusable task.
- The original initializer captures the actual App, ChaChaNotes owner/path and marks-service owner before construction. Publication must still belong to that same owner after native construction, and every newly created refused initialization connection retires on its original source thread. Existing service/native borrowers are not adopted or closed.
- The initial startup task holds one finite child task; repeated cancellation drains the same actual callback before propagating cancellation or releasing startup custody. Normal exception stays visible, and cancellation does not publish a screen. Runtime disposal closes admission first and serializes with the original initializer lock; no disposed Runtime publishes storage.
- After actual callback retirement, the initial route rechecks exact App/Runtime, startup task/loop/thread, profile/source, screen stack, current tab, and shutdown/initial-push state before the original screen construction and push. A newer destination is never overwritten by a stale initial push.
- Live first Send retains every original bridge, storage, permission, provider, capture and readiness guard. Receipt preparation makes no claim of permission readiness. All original startup, Send, heartbeat, helper and native-open limits remain unchanged.

Native qualification plan: retain the original cold held-query causal RED first; then real queued reader replacement and same-function code/default drift, in-body App/DB/marks/generation drift, repeated cancellation while actual native SQL is held, disposal during the same held callback, warm borrowed original connection, preinstalled custom/instance/subclass/memory fallbacks, and normal first composition/key/Send/disposal controls. Controls observe original code/native handles; no native calls, guards, waits or budgets are replaced.

Evidence-only candidate is not installed or Native-qualified. Root owns task/plan/ADR registration and serialized actual runs.

Implementation order: retain original Native6 causal RED and source receipts; independently review the frozen v2 two-source patch against exact installed bases; install only after these ADR085/126 amendments; verify20 real native finite-callback/source/owner/custom/borrowed/cancellation/disposal controls using the corrected original admitted-native qualifier; re-run unchanged cold composition/key/disposal control and original whole/platform evidence. Live UAT on ded197 passes all3 real captures/replies/links; repeat final-source setup/first Send if later production changes touch that route. No original deadline, guard, provider/capture policy or startup-ready publication is relaxed.


## Stock agent-turn admission work-count investigation (before implementation)

- [ ] The exact stock source-qualified run-turn callback retains fresh admission for every captured owner/path and its already captured log root without redundant execution-scope entries inside that same finite callback. Existing worker/model-loop guards, custom/legacy/scoped-none paths and all permission/retirement policies remain.
- [ ] Actual original-source/native owner-path counts qualify before any edit; root/method/source/pause drift and custom fallback controls preserve refusal and physical custody. Original whole limits determine acceptance after any repair.

ADR required: no new ADR for a same-callback grouping if it preserves the existing admission contract; existing ADR126 applies. Reassess before implementation if the actual controls require changing a runtime/authority boundary. The current source report identifies nested Agents/activation.py execution entries that re-enter every active scope when adding only the captured stock log root. Source repetition alone does not establish current cost. Prepare narrow original-effect Native controls with exact sources/actors/physical retirement and currentness refusals; retain custom/scoped-none resolver paths; do not rewrite the general guard, cache authority, reuse values over await, or raise budgets. The real live pre-reservation gaps are elapsed7.813/5.578/5.000s, without exclusive helper attribution. Implementation remains pending causal controls and source review.

## Workspace composite verification checkpoint

Implementation Notes: cold Consent admit/status, Inspector capture and registry save-binding now share one original counted connection interval. Exact newly opened handle retirement replaces only the new outer lazy-helper scope; foreign cache replacements and original borrowers remain live. Six candidate failures were reproduced before the captured cleanup correction. All21 actual native controls pass104.563s with current sources; the three new test files pass format/lint, and differential static checks over249 changed/new Python files introduce zero diagnostics (735 existing on dev,734 current). Original service modules retain79PASS7SKIP18 InspectorFAIL on Windows; original no-follow descriptor routines are AST-identical, and actual stdlib lacks their required flags. Broad native/startup/helper gates and final PR-source platform qualification remain pending; status stays In Progress.


## Stock run-turn finite admission refinement (before production)

ADR required: no new ADR; existing ADR126 finite same-callback source custody applies.

The corrected passive original-code native observer records nine execution-scope starts for five distinct actual owner/path keys inside one stock guarded run_turn. It observes the original contextmanager generator without replacing production callbacks. The same source-qualified run_turn finishes, its native worker SQLite handle physically closes, storage/monitoring retire, and unchanged source checks pass before the work-count assertion fails. All five custom-selector/root/method/source/pause controls pass. Native red-1 is excluded for the callback adapter's one-frame observer mistake; red-2 is the genuine causal result,44.015seconds. This establishes duplicate work, not exclusive time savings.

For only an already captured qualified stock scoped log source, append its agents.history root to the first original execution source set. Keep the original scoped_log_source qualification and body within that single original execution context. Remove only the second whole-source-set execution entry from this stock path. No authority is reused beyond this synchronous callback. The unqualified scoped=None selector still executes inside its preceding first admission and retains its second fresh log-root admission; independent worker/model guards, generic guard code, native owners and permission/pause policies remain.

Acceptance: the same six native controls must pass with one scope start per actual stock owner/path; the custom selector must retain its original arg-free call and duplicate set route, all retarget/pause controls must refuse provider/log effects and retire resources. Then run the appropriate original scoped log/agent activation compatibility nodes and unchanged whole/platform limits. No tests, bounds, source qualifiers or coordinator checks are relaxed.


### Original cold startup completion prerequisite (before test edit)

ADR required: no new ADR; test-only synchronization implements the registered asynchronous initial-screen contract in ADR085/126.

The first v3 original cold receipt qualifies the startup-issued native worker, exact original handle/lease retirement and actual loop progress, but its body takes app.screen before the async initial push completes. Observe only the original `_initial_screen_pushed` latch before taking the screen, within the existing enclosing asyncio.wait_for240 and unchanged child240 deadlines. App sets that latch after its original push_screen and current-tab update. Keep the subsequent exact ChatScreen, composer/key and all source/native/lease/cleanup assertions unchanged, so a wrong completed destination still fails. The only added while/sleep prerequisite is separately recorded; removing it yields the exact prior script AST. Native6 remains the original causal RED and v3 original cold1 remains a prerequisite failure until a fresh run. No product change or performance-limit increase.


### Accepted native stock/startup checkpoint

Implementation Notes: stock admission grouping passes6/6 native controls51.000s and all62 appropriate original activation/scoped-log/body-binding/service-wiring tests380.625s, with source unchanged. The original cold Console control passes53.750s after the separately reviewed completion-latch prerequisite, retaining its actual original SQL hold, source/currentness, typing, disposal, physical handle/lease retirement and all original assertions/deadlines. Its worker is issued by the actual original initial task and loop; UI progress occurs while the original admitted native SQL is held. The24 startup native custom/binding/owner/source/cancellation/disposal leaves already pass. No whole-startup/Send/heartbeat/helper/native-open acceptance is inferred from these controls. Census exceptional generator accounting remains an observer prerequisite under repair; status stays In Progress.


Implementation Notes (accepted observer checkpoint): original-code census preserves stock callback identity; eight original credential ticks pass with628 fully retired spans and unchanged limits. The Windows full census retains its original POSIX helper anti-vacuity failure; no whole-budget acceptance or zero-native-I/O claim follows. QA records the exact ded197 macOS/Linux source-qualified whole helper failures and actual callback partitions. Status remains In Progress; final dev integration and whole/live/platform verification are pending.
