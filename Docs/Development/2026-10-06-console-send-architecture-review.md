# Console Send architecture direction review

Status: historical working notes consolidated into the [written specification](../superpowers/specs/2026-10-06-console-send-preparation-architecture-design.md) and [ADR-225](../../backlog/decisions/225-console-send-preparation-and-io-ownership.md). Written review and the requested issue audit are complete; proposed-section labels below record the discussion sequence. No implementation or performance acceptance is claimed.

Task: TASK-34563. User approved the coordinated phase direction and requested this review before continuing. Review used the current Console sources in the integration worktree at HEAD d382e1cda1ede77a49da1c64391d2a36c53d6c4d, including its working-tree source. Earlier latency samples belong to their recorded bases and do not qualify this newer source.

## Assessment

Retain the existing runtime, controller/store authorities, preparation state and provider-request models. A thin, explicitly ordered coordinator and domain-owned I/O are the preferred direction. Moving bodies into new files without removing repeated work would not meet the goal. The design needs the following refinements before it can be implemented.

## Required refinements

### 1. Separate receipt, runtime custody and durable acceptance

The current UI wiring awaits `capture_turn_configuration_snapshot` before constructing `ConsoleTurnCustodyRequest` and transferring runtime custody (`UI/Console_Modules/wiring.py`, 349-475). The runtime's `accept_turn` requires that complete configuration and transfers attachments before scheduling its task (`Chat/console_runtime.py`, 2450-2509). Therefore a promise of early runtime ownership cannot be implemented by moving the existing call alone.

Define a small, bounded received-intent type for the exact draft, selected session and input revisions; promote it to the existing fully prepared request only after checked capture succeeds. Runtime custody of this intent must not grant execution or claim durable acceptance. Use the existing lifetime owner, not another job registry. Define whether edits during preparation invalidate the attempt according to the existing preserved-draft contract; cancellation/refusal must never overwrite a newer draft or consume another attempt's attachments. Ordinary queue availability stays tied to durable acceptance. Navigation and shutdown behavior before and after custody must be explicit.

### 2. Make phases responsibility boundaries, preserving dependency order

Postcommit work currently includes identity/settings publication, durable owner publication, staged-input clearing, workspace projection, queue acknowledgement, accepted callbacks, prompt history, preparation publication and a fresh hook admission check (`Chat/console_chat_controller.py`, 13002-13143). Several depend on durable identities or recovery state.

Do not move all provider preparation before commit, run every stage in parallel, or treat all postcommit work as optional. Classify each effect by its required input, durable obligation, ordering, failure and retry behavior. Preserve current order initially; move work only when its own reviewed contract permits it. UI feedback must distinguish received/preparing, awaiting approval, durably accepted and dispatch started.

### 3. Specify shared data freshness independently of authority

A preparation result may share bounded data for one identified attempt. It may not retain a permission verdict, credential grant, native lease or root-bound handler for later attempts. Configuration choices, catalog definitions, mutable consent and credentials have different freshness rules.

Record the exact owner/source and relevant revisions. Reject stale publication after a source/session/root change or approval pause, and bound any rebuild rather than loop indefinitely. Preserve existing permission checks at actual execution. Cross-store coherence needs declared checks; independently captured stores are not one atomic snapshot. File watchers or time-based caches cannot establish authority.

### 4. Keep native I/O ownership inside its storage domain

Batch related operations within an existing finite worker/transaction where their actual owner and lock order allow it. Never carry a native proof across an await, invoke arbitrary extension callbacks under a storage lock, or merge hook, MCP and SQLite owners under one lock. Returned results contain detached data. Stop and repeated cancellation must drain issued native work and reject late publication before creator resources close.

### 5. Preserve reconciliation and recovery effects

Hook reconciliation is not just a read: disable transitions rotate grant tokens and removals retire grants (`Agents/hook_permissions.py`, 260-316). MCP definition changes may also persist a downgrade and an emit-once audit. An unchanged or disabled feature must retain its required history effects.

The existing postcommit claim/completion mechanism marks an effect only after success and releases claims on failure (`Chat/console_chat_controller.py`, 11642-11695). Retain that owner and the dispatch checkpoint/retry boundary. No global exactly-once claim is possible across independent stores or an uncertain remote provider request. Moving an auxiliary write after dispatch needs an explicit ordering, drain, failure and recovery policy; prompt-history writes are currently awaited, serialized and cancellation-aware.

### 6. Keep coordination and extension contracts narrow

The coordinator sequences work; it does not become a second state machine, permission engine, database owner or broad controller proxy. Existing state remains authoritative. Give domain operations named inputs/results and explicit dependencies; avoid a generic step graph or dependency bag. Known supported callbacks retain their live lookup and signature contracts. Normal private relocation follows ADR-220's documented compatibility limit rather than accumulating wrappers for arbitrary private monkeypatches.

A future feature should identify the phase it affects, the data it needs, its effects/freshness/failure rules and its work budget. The design should show a concrete new-feature example to prove that this is usable without editing unrelated stages.

### 7. Make concurrency and feedback qualification explicit

Retain per-session ordering, existing queue/global-cap behavior and bounded native concurrency. One slow preparation must not block another chat's input or acquire a profile-wide global execution lock. Duplicate Send, source changes during approval, Close, Stop and shutdown need defined outcomes at each boundary.

The 100 ms requirement applies to actual rendered receipt and input responsiveness, not a status assignment, direct action call or task creation. Verify the real Enter/button paths, including eager task scheduling. Separate adapter-entry overhead from network, deliberate approval waits, retrieval and cold initialization; those routes still require responsive feedback. Use scoped original read/admission/write counts plus minimally instrumented end-to-end timings on supported hosts. The one-second ordinary dispatch goal remains a target, not an architectural guarantee.

## Next design section

Define the lifetime of each shared preparation result and the owner of every required I/O/effect. Include a freshness table, the required dependency order, and concrete cancellation/retry examples before writing the final specification. Existing ADR-094, ADR-098, ADR-126, ADR-197 and ADR-220 constrain this design; the chosen interface/lifetime changes need a recorded architectural decision before product implementation.


## Proposed section 2: shared results, freshness and I/O ownership

Status: section direction approved subject to the requested review; the review refinements below are incorporated. This remains a working design, not the final specification.

### Result lifetimes

| Result | Owner and consumers | Lifetime and freshness rule |
| --- | --- | --- |
| Received input | Existing runtime custody owner; preparation coordinator | Exact bounded draft, selected session, input revisions and selected in-memory choices. No file access, credentials or executable permission. Promotion, refusal and cleanup refer to this exact request. |
| Frozen turn configuration | Existing turn configuration/context owner; provider/request builders | Captured through checked preparation for this attempt. Holds selected values and narrowing maxima. Retry creates a new attempt; existing durable retry policy decides what inputs remain fixed. Navigation alone does not invalidate app-owned work. |
| Tool preparation data | Existing MCP/control-plane owner; MCP and local catalog builders | One live composition operation after its prerequisites. Shared kill-switch observation, catalog records and policy projection are data for building the same attempt's catalog. They do not authorize execution. Provider handlers remain run-owned. |
| Hook preparation outcome | Existing hook consent owner; coordinator and review projection | One reconciled configuration/store observation with its existing revisions. Review resumes re-enter the owner; actual launches retain current definition/consent serialization. No cached clearance survives revocation. |
| Commit result and effect completion | Existing store/repository and postcommit effect owner | Existing durable identity/checkpoint rules determine recovery and retry. No second coordinator ledger or cross-database transaction. |
| Display projection | Existing presentation owner; mounted views | Derived data for rendering. It cannot authorize Send or trigger native bookkeeping while being read. |

Each typed result has an attempt/session identity and only its relevant source/owner and observation fields. Source identity is not content identity, and an observation is not a universal permission revision. Use actual domain revisions or policy identities only for the fields they cover: the existing MCP profile policy digest excludes the global kill switch and lifecycle metadata, and legacy profiles have no imported-profile revision. Use domain-specific types and existing constructors; do not build a generic revision registry or one giant context bag. Immutable results cannot retain widgets, open connections, native leases or credential payloads as reusable preparation data. Runtime-owned resources remain with their existing owners. Nested schemas, records and mappings must be detached and immutable to consumers; a frozen outer dataclass is insufficient. Normalize/freeze once at the owner boundary, with explicit size bounds, and share the result rather than deep-copying it for every builder. Avoid returning full sensitive configuration/store payloads when only a normalized projection is needed.

### Reuse boundaries

A successful domain operation produces one result that its pure consumers share. For example, ordinary MCP/local catalog preparation derives the compose-time switch and MCP policy projection from the same permission observation, rather than asking each builder to load it independently. The resolver already accepts a loaded payload. Catalog/profile observations and downgrade/audit effects remain owned by the control plane. The returned result must state the effective data and outcome of those effects, rather than ambiguously mixing a pre-write snapshot with post-write metadata. Use the existing mutation fence or precise compare-and-set for dependent read/modify/write work, preserving edits within the protection actually supplied by that owner. Do not infer cross-process serialization from a process-local fence. Retain the existing blocking versus best-effort effect semantics; audit failure must not silently become a new dispatch prerequisite.

The initial frozen definition maximum and the later live composition have different jobs. Keep the former as a narrowing ceiling; a later observation can remove eligible tools or refuse the result, but cannot silently expand the submitted maximum. Do not promise one permission read for an entire Send: admission, resumed approval and actual execution are distinct boundaries. Consolidate repeated reads within a boundary; retain reads required by an intervening change or effect.

A source/session replacement or relevant input revision change rejects affected publication. A Settings change must not silently substitute a new provider or workspace into a frozen request. Approval resume revalidates the relevant inputs and authority, using the existing review/retry outcome rather than granting clearance from the old result. Reuse unrelated pure results only if their own dependencies remain valid. No unbounded automatic rebuild loop, file-watcher authority, TTL permission cache or profile-wide generation that invalidates all results for every small change.

### Native operation shape

The coordinator requests complete domain operations with declared inputs/results. It does not open files or manage transactions itself. A finite worker performs the needed reads and required related writes under its domain's existing admission and locking rules, then returns detached data after its native resources retire.

Batch declared related file targets and avoid helper-owned rediscovery where an equivalent same-operation contract can be established. Preserve per-target path/identity/pause checks and exact canonical source selection. Existing producer recovery observations remain independent where their authority contract requires them. A common profile does not make catalog, permission, hook and SQLite owners interchangeable.

Preserve lock order and existing thread/loop affinity. Pure inventory callbacks that belong on the app loop stay there; arbitrary callbacks do not run under storage locks. Each native worker segment contains no await. One semantic preparation API may use multiple loop/worker segments where the existing callback affinity or effect order requires them; fewer helper-owned reads does not imply one thread or one giant lock for every route. Cancellation retains custody until the issued native work actually retires, and then rejects obsolete publication. File-store changes and SQLite commits remain separate owned effects with defined failure behavior, not a claimed global atomic commit.

### Execution and extension

Tool invocation, hook launch, credential resolution, workspace access and durable mutations retain the live gates owned by those services. Builders consume preparation data through an explicit API; they do not obtain authority from an internal supplied-data flag. Public/custom adapters keep their documented late lookup and call shape until explicitly migrated to the new contract. Shared stock data must not bypass a custom receiver or reader; unsupported adapters retain their ordinary route. Validation of detached data is not a supplied-data authority bypass.

A future tool category adds its domain capture/result and pure catalog construction to the existing tool preparation boundary. It supplies its live invocation gate and required effect rules. It does not teach the UI, controller, history service and every provider builder to load the new policy file independently. The implementation plan must include a small concrete extension example and a measured I/O count to prove this boundary works.

Remaining design sections: receipt/approval/cancellation/retry state behavior, required effect dependency order, compatibility/migration and targeted performance qualification. Product implementation awaits the reviewed final specification and implementation plan.


## Section 2 review findings and refinements

Review basis: the current MCP permission load/mutation, catalog composition and invocation paths, hook snapshots and the source-owned worker contracts. No product code or runtime verification was performed for this design review.

1. **Revision coverage must be explicit.** `_CapturedSources` identity receipts identify receivers/bindings; they do not establish that policy bytes remain unchanged. `profile_policy_digest` covers selected profile policy, not the global kill switch or all inherited profiles. A profile revision is present for imported profiles, not every legacy profile. A returned observation remains data; externally mutable authority is observed again at its actual authority boundary. Do not create a generic revision stamp with stronger claims than its inputs support.
2. **Read/modify/write must preserve concurrent edits.** Effective-state resolution can trigger individual `mark_config_changed` mutations that reload the store. Consolidation must not compute from one payload and later overwrite a newer payload with it. Use the owner fence or field-appropriate CAS, preserve other edits, and define the effective returned projection after the operation's own changes. Audit logging remains separate, with its existing best-effort behavior and emit-once policy. Do not claim a joint permission/log transaction.
3. **Error results must retain their meaning.** Permission `load` treats absence as defaults, corruption as its existing backup/reset behavior, and uncertain I/O/admission failure as an exception. The new domain result must preserve these outcomes; unavailable input is not an empty catalog or a disabled feature. Domain outcome metadata should come from the actual read/effect, without extra reads merely to manufacture status fields. Publish no incomplete successful result.
4. **Live-check error policy needs an explicit follow-up.** Composition rejects failed switch reads, while some existing invocation helpers treat a failed switch read as not engaged. Approved-call stamps and other gates also affect the actual execution path. Therefore the phrase live permission checks is not a proven universal deny-on-uncertainty guarantee. The lifecycle/error section must specify current per-route behavior and any deliberately approved hardening, supported by focused authority tests; this consolidation must not silently redefine it.
5. **Pure consumers should stay pure.** Validate capture and owner/attempt applicability at completion/publication, at a resumed review boundary, and at actual authority use. A pure builder must not trigger another native freshness scan for each field it reads. This does not remove required native checks inside the actual domain operation. No TTL/watch-based clearance or new general invalidation framework is needed.
6. **Shared ownership requires deep immutability and cleanup.** Bound and freeze nested result data once; do not share mutable dictionaries or root-bound handlers among attempts or chats. Retire results when their owning attempt releases them. Late cancellation outcomes may preserve evidence of an already completed write, but cannot publish provider state into an obsolete attempt. A semantic domain operation can have multiple finite worker segments when affinity requires it; retain the physical drain of each one.

Required examples for the final specification: another process flips the global switch during preparation; a profile changes while an approval is open; a definition changes after the submitted maximum is frozen; a downgrade mutation races a profile edit; an audit write fails; a permission read fails; a consumer tries to mutate a shared schema; cancellation arrives after a native write but before publication. These are focused verification cases, not a proposed full-suite sweep.


## Proposed section 3: receipt, approval, cancellation, retry and errors

Status: proposed for review. The user authorized development of this section and continuation through migration/verification. The requested unknown-authority policy is an explicit design change from existing best-effort switch-reader exceptions; no product change has been made.

### Lifecycle ownership and visible outcomes

Use the existing runtime custody owner and controller/store preparation and run state. The following rows describe boundaries and outcomes, not another runtime state machine. The new received-intent variant is not a partially valid `ConsoleTurnCustodyRequest`; promote it into that complete existing type only after checked preparation succeeds.

| Boundary or event | Required behavior |
| --- | --- |
| Receive Send | Validate bounded in-memory input and extend the controller's existing per-session admission boundary to reserve this received intent; the complete preparation model is constructed later, not with missing configuration fields. Register the exact intent under runtime custody before scheduling slow work. Publish a Preparing projection immediately; do not claim a saved or dispatched turn. No credentials, native read, heavyweight import or full configuration copy precedes this receipt path. Schedule presentation before slow work and explicitly yield in the received job before preparation so eager task creation cannot execute blocking setup inline. Rendered receipt remains a measured contract, not an assumption that one yield guarantees paint, and headless work does not wait for a nonexistent view frame. |
| Receipt scheduling failure | Release the exact reservation/custody and leave the draft, staged inputs and attachment ownership intact. No request may be stranded between registration and task creation. |
| Repeated Send during preparation | Use the existing per-session refusal; do not submit the same request twice or enable ordinary queueing before its accepted boundary. Preserve existing queue/cap behavior after acceptance. |
| Checked capture and hook review | Preserve the exact request and owning session. A relevant draft, attachment, evidence, settings/source or hook-definition change before promotion refuses the obsolete attempt; preserve newer input. Configured hooks still reconcile their observed history even when presently disabled. |
| Approval pause | The existing runtime-owned decision host retains the pending request; mounted views only present and resolve it. Register hook-consent presentation through a named kind-specific bridge rather than retaining a screen dispatch closure. No native locks or leases remain open while waiting. Hidden decisions use existing attention/remount and answerable-time rules. |
| Approval answer | Atomically consume the exact pending continuation once. Approval is not permission to skip fresh capture: re-enter the relevant authority owner, validate source/definition/profile/root applicability, and refresh affected data. An obsolete, denied or cancelled answer cannot revive Stop/Close or execute a newer draft. |
| Precommit refusal/error/Stop | Seal this attempt against commit/provider entry, retire its issued work, and preserve or offer its exact draft through the existing recovery owner. Never overwrite newer composer text or silently resubmit recovered input. |
| Commit in flight | Retain the exact native future through repeated cancellation until it has committed or rolled back, or until its owner records unresolved native uncertainty under existing recovery rules. Stop fences provider entry. A cancelled await is not evidence of rollback. Reconcile the actual outcome before returning a draft to retry or releasing its admission. |
| Durable acceptance | Publish the existing commit identities, consume only the matching staged inputs and acknowledge its queue claim through the existing postcommit effect mechanism. Ordinary queue availability and accepted callbacks remain tied to this boundary. A received intent is not accepted merely because the UI painted Preparing. |
| Accepted but not dispatched | Retain the existing accepted checkpoint, required postcommit effect order and trace/provenance requirements. Failure here enters the existing recovery/refusal path rather than creating another user message or losing the committed turn. |
| Provider entry | Perform the existing dispatch-start checkpoint/trace boundary and required live authority checks in their established order. Do not move checkpoint creation earlier merely to hide slow work; preserve its existing uncertainty semantics. |
| Stop after acceptance/entry | Stop the selected turn/chain using current terminalization and recovery owners. Preserve the durable commit and any existing partial response. Settle native work before releasing resources or allowing a racing attempt to replace its owner. |
| Navigation and another chat | Detach only the view. Runtime-owned work and pending decisions remain attached to the original session. Prepare/build from its frozen input and app-owned services, never a newly active widget. A successor view cannot be cleared or updated by an obsolete callback. |
| Session Close or app shutdown | Existing confirmed close/quit fences admission first, prevents new preparation/approval resumes, then cancels and drains owned work. Retain disposed owners so timers cannot reconstruct fresh services. Existing bounded shutdown and unresolved-resource evidence remain authoritative; an expired deadline cannot claim physical retirement. |

Receipt/custody promotion defines draft handling precisely. While the received intent is still checking the captured input or awaiting initial hook review, an edit revision invalidates that old attempt under the existing frozen-review contract. After full request promotion, a new composer draft is independent of the request already under custody. Completion, refusal and recovery may clear/restore only the captured revision or expose its separate existing recovery entry; they never replace newer text. Transfer attachment/evidence custody once at the explicit promotion boundary, with unwind on failure. No private input is persisted just because a receipt was displayed.

The coordinator must not retain widgets or screen-bound callbacks after handoff. A lightweight intent includes detached selected values, stable identities/revisions and staged-input references only. App-owned providers finish checked configuration capture. View attachment/generation belongs to presentation, while request/session identity belongs to the surviving job. Manual button/Enter, spoken Send, queued turns, agent starts, wakes and dispatch recovery join the appropriate existing domain boundary without inventing a fake UI receipt for headless work. Slash commands and raw-console routes retain their own dispatch/refusal semantics.

### Unknown authority and error policy

At an authority-dependent operation, represent a failed required observation explicitly as unavailable/unknown. It is never coerced to false, an empty policy, permission disabled or an old successful value. Refuse the relevant operation before its effect, even when a previous approval exists. This intentionally hardens the current invocation helper paths that can treat failed kill-switch reads as off. It does not make every optional feature or diagnostic failure fatal to an ordinary chat.

| Failure or observation | Outcome |
| --- | --- |
| Required tool permission/kill-switch/workspace authority unavailable | Refuse that tool operation before execution; return a neutral unavailable reason. A prior approval stamp cannot establish a failed independent live check. Do not claim the user denied it. |
| Hook configuration or consent unavailable | Do not launch the hook. Retain the shared admission/review refusal required by the existing hook contract. Authority failure cannot inherit UserPromptSubmit's execution-error fail-open policy. |
| Hook execution failure after authority was established | Preserve the event's existing deny/fail-open/timeout behavior, argv-only protocol and process-group ownership. |
| Optional MCP preparation unavailable | Preserve the caller's documented no-MCP/fallback behavior where allowed. Do not fabricate a successful empty observation, silently broaden the submitted maximum, or weaken a request path that explicitly requires the capability. No new Send-without-tools mode is introduced. |
| Missing/corrupt policy file | Use the domain owner's existing defaults and backup/recovery rules, with their actual outcome recorded without extra status reads. Uncertain I/O remains an error, not corruption. |
| Destination/Library authority change, required persistence or trace failure | Preserve the existing refusal, interactive pause/bypass choices, autonomous refusal and retry contracts. No provider call precedes required durable work. |
| Best-effort audit/capture/presentation failure | Preserve execution policy and committed state; expose the existing content-free degradation. Do not make an audit write a new required permission transaction. Required trace provenance/call admission remains separate from best-effort capture. |
| Cancelled/late result | Drain its actual issued work, preserve evidence of already completed durable effects, and reject stale publication. Do not replace the original error with cleanup bookkeeping failure. |

### Retry and effect order

Precommit Retry creates a new attempt and re-enters the necessary captures. Existing held-preparation actions determine which selected inputs are preserved or deliberately recaptured; retry does not automatically replay old hook side effects, consume another staged input or widen the frozen maximum.

Postcommit Retry uses the existing assistant/user identities and checkpoint. Re-enter `retry_dispatch_recovery`, including destination and frozen-authority compatibility, fresh provider resolution and the existing context/retrieval rebuild. The coordinator cannot turn this into a new Send. A provider-started/uncertain attempt retains the current conservative recovery path; no automatic network replay or global exactly-once guarantee is introduced.

Initially retain this existing required postcommit sequence: identity/settings and roleplay reconciliation; durable message-owner publication; exact staged-input clearing; workspace projection; queue acknowledgement; accepted hook/context handoff and manual accepted callback; ordered prompt-history append; preparation publication; fresh hook admission; existing trace/dispatch/provider boundaries. The label accepted_hook includes provider-message handoff, so it is not merely a disposable UI notification. Best-effort return/failure behavior remains owned by each callback. History remains awaited in this design; moving it after provider entry would require a separate ordering/recovery decision, not an untracked background task.

Use existing effect claims/completion and checkpoint transitions for repeated resumes. Record completion only when the effect's existing success contract is met. A coordination refactor must not wrap a whole sequence as one opaque completed effect or claim atomicity across independent stores.

### Native ownership refinement found during this design

Current `AgentChatStart` commit code retains and shields an explicit commit worker; ordinary manual commit can call the naked `_run_durable_db_call` to-thread await. The architecture must require explicit custody of every issued native future, including the ordinary commit path, with registration before await and physical outcome settlement before recovery/creator close. This is a lifecycle correction to verify, not a claim that existing task cancellation proves native rollback. Reuse the actual runtime/callback custody owners and per-attempt references; do not add a general task framework.

Current MCP `mutation_fence` is a process-local path RLock. It cannot establish cross-process write serialization, and an atomic file replacement is not by itself an atomic compare-and-set. The preparation refactor retains actual store mutation contracts; any stronger cross-process writer contract needs a separately explicit storage-owner decision and qualification. Cross-process revocation observations are still required at their existing live gates. This limitation must not be obscured by a generic snapshot revision or a claim of a transaction over several JSON files.

## Proposed section 4: extension, migration and verification

Status: proposed for review together with section 3. These are delivery boundaries and evidence requirements, not an implementation plan or a promise that latency targets have already passed.

### Extension contract and concrete example

A new read-only workspace tool family registers its supported tool IDs and builders through the existing tool catalog provider seam. It consumes this attempt's selected roots, policy projection and immutable definitions from tool preparation. The family implements its own live path/exclusion/access gate and resource ownership. If it needs another source, its domain capture declares that source, freshness scope, required effects and error outcome. It does not add a settings read to every provider, change prompt-history ownership or acquire an unrelated hook lock. The builder is independently testable with detached data; actual execution is independently testable against the real workspace authority.

Dependencies remain explicit and named. Stock prepared-data APIs do not bypass custom readers or callbacks. Preserve known public signatures and late lookups with narrow adapters during migration; migrate supported adapters deliberately. Normal private relocation uses ADR-220's existing compatibility limit. No reflected dependency bag, generic step graph, new scheduler, cache TTL, duplicated state ledger or configurable public legacy/fast mode is needed.

### Migration boundaries

1. **Receipt and custody:** introduce bounded intent/promotion and screen-free capture behind the current entry points. Preserve draft/attachment/queue semantics and prove real input responsiveness. Keep final domain authorities and durable ordering unchanged.
2. **Tool domain:** add one explicit preparation API/result used by the stock MCP and local composition consumers, with nested immutability, covered observations and required mutation/audit outcomes. Preserve the initial maximum, live invocation gates and custom fallback contracts. Migrate both live and disposable preview consumers with their existing publication distinctions.
3. **Hook domain and approval:** reuse the actual consent/reconciliation owner and add the named runtime decision bridge. Consolidate same-boundary reads without removing consent-history transitions, final launch checks or event-specific error rules.
4. **Lifecycle/effects:** unify ownership of issued native work, including ordinary commit; retain current checkpoints and effect order. Move only the preparation orchestration into its focused owner, removing each migrated duplicate path rather than retaining two long-lived pipelines. Keep history awaited.
5. **Qualification and cleanup:** prove the final integrated source, remove temporary adapters that no supported caller still needs, record remaining bottlenecks, and update task/ADR/docs. Unproven micro-optimizations are not a prerequisite; reassess the earlier uncommitted MCP-member candidate under the new contract rather than automatically carrying it forward.

These boundaries can be separate atomic implementation tasks after written-spec and plan approval. Each migration must cover the relevant manual, queued, background and retry consumers rather than qualifying a fast path while leaving alternate entry points unsafe. The design task references only existing tasks; future implementation tasks will be created in dependency order after the design is reviewed.

### Evidence and performance requirements

Reuse existing focused preparation, runtime lifetime/shutdown, hook review, queue, dispatch recovery, MCP read-error/snapshot, native storage and cancellation controls. Add checks for the new intent/promotion, unknown-authority refusal before effects, single-consumer continuation, deep immutability, narrowed maxima, callback compatibility and the explicit native commit future. Important interleavings include duplicate Send, a new draft while approval is open, source/profile/global-switch changes, a raced downgrade write, Stop during commit, repeated cancellation, Close during publication, navigation/remount, and late worker success/error. Preserve original assertions, physical ownership evidence and existing deadlines; fixtures or direct action calls alone do not qualify the product.

Count original file loads, native admissions, path checks, writes and worker handoffs by domain and operation. Verify that a new pure consumer of the shared result adds zero native reads/admissions. Bound nested-result size and avoid repeated freeze/copy work. Distinguish necessary new observations after an authority boundary from repeated helper work within the same operation. Do not add global monitoring or permanent detailed diagnostics to the production hot path.

For final latency, drive actual Enter and button events through the real app pumps (including eager scheduling) and observe rendered Preparing/approval/refusal/error frames and input responsiveness while a real native operation is held. Record event-to-rendered-receipt and event-to-actual-adapter entry with one monotonic clock. Require receipt/input feedback within 100 ms and ordinary application overhead under one second; separately report cold initialization, enabled features, deliberate approval waits, retrieval and provider/network time. The time exclusions do not exempt those routes from immediate feedback. If the first slice misses the adapter target, report the gap and attribute the remaining work rather than redefine completion.

Run matching private baseline/candidate comparisons with the same source qualification, interpreter/dependencies, filesystem placement, feature shape and minimal instrumentation. Alternate run order and retain individual cold/warm samples. Require real persisted replies, complete trace links, correct checkpoint settlement and positive native/process retirement; forced termination or source drift invalidates a performance claim. Qualify Windows, Linux and macOS separately before claiming supported-host acceptance. Run targeted tests only; the full suite remains opt-in.

### Architectural decision and review handoff

ADR required: yes. The final architectural decision must record lightweight receipt/custody promotion, domain preparation/result contracts, observation versus authority lifetimes, intentional unknown-authority hardening, native-work outcome ownership, existing effect/recovery order and compatibility limits. It extends ADR-094/098/126/197/220 and the relevant hook/tool authority decisions; it does not silently supersede their storage or runtime contracts. Allocate and record the canonical decision before any product implementation or implementation-plan execution.

After approval of the remaining in-chat sections, consolidate these decisions into one self-reviewed specification under `Docs/superpowers/specs`, link the canonical ADR and design task, and commit only those reviewed design artifacts. The written specification is then reviewed before invoking writing-plans. Existing uncommitted diagnostic and production candidates remain separate and unclaimed as complete.


## Remaining-section self-review

- Received-intent custody is distinct from complete request promotion and durable acceptance; the existing preparation constructor is never given missing configuration.
- The new early admission reservation extends the existing session boundary rather than creating a parallel queue/state authority.
- Invalidation before promotion and independent new drafts after promotion are explicitly separated; source/draft callbacks cannot overwrite successor input.
- Approval host ownership has an explicit hook-consent bridge; no surviving screen dispatch closure or native lease is retained across the wait.
- Commit cancellation records and settles the actual native future; a late commit cannot be treated as an uncommitted retry merely because its await was cancelled.
- Unknown required authority refuses the affected operation before effects; optional fallback and diagnostic policies remain separate. This hardening is intentional and requires focused compatibility/authority controls.
- Process-local mutation fences and field-scoped digests are not misrepresented as cross-process transactions or complete policy revisions.
- Required postcommit ordering, trace requirements, checkpoint uncertainty and current recovery entry points remain explicit. History is still awaited.
- Receipt scheduling accounts for eager tasks and headless work; actual input/render evidence is required before claiming the 100 ms target.
- Migration and evidence are reviewable design boundaries only; no implementation plan, product change, completion claim or full test sweep has been performed by this design stage.


## Consolidated written-spec review

The user reviewed the written specification and requested an issue audit before planning. The audit retained the architecture and corrected these implementation-readiness gaps in the spec and ADR:

- Store/controller received admission and complete promotion share one atomic claim across entry points; runtime remains lifetime ownership. Cleanup is generation-specific, and archive/Close guards include received work without claiming it is saved.
- The runtime hook-review bridge and native outcome custody are prerequisites to early-receipt enablement. Hook read consolidation can follow later; approval ownership cannot.
- Tool maxima are attempt-scoped. Current checkpoints persist Library/destination authority, not the old MCP maximum, so restart follows existing fresh recovery rather than a fictitious preserved ceiling. Stronger persistence is outside this no-format-change design.
- Domain acceptance and UI clear effects are distinct. Clearing must compare input revision and view generation, including identical text retyped later. Complete-request attachment-prefix compatibility is retained separately from the strict received-intent capture rule.
- Unknown-authority hardening covers upstream stock wrappers and approved-call paths; a local closure currently converts switch errors to false before the provider sees them. Demand-driven capture avoids unused catalogs/providers and does not reuse live data as preview publication.
- Performance evidence reports raw elapsed samples. Send-triggered setup, native checks, worker queues, history and trace preparation remain application cost; phase labels do not remove it.

Source checks: ConsoleChatStore.begin_preparation accepts a complete preparation; Runtime._register_custody counts requests by turn ID and archive owner; attachment transfer retains an exact prefix; the checkpoint contains no MCP maximum; retry captures fresh configuration; the controller-injected local switch closure converts read errors to false. These are verified source facts, not runtime bug or latency acceptance claims.

Document validation and whitespace checks are run before the correction commit. Runtime verification remains part of the implementation plan; no full suite or product changes were made by this review.
