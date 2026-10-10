# Console Send preparation and I/O ownership

Date: 2026-10-06
Status: Accepted following written review and the requested issue audit; implementation planning is next.
Task: [TASK-34563](../../../backlog/tasks/task-34563%20-%20Design-Console-Send-preparation-and-I-O-ownership.md)
Decision: [ADR-225](../../../backlog/decisions/225-console-send-preparation-and-io-ownership.md)
Source review: [Architecture review](../../Development/2026-10-06-console-send-architecture-review.md)

ADR required: yes.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: received-request custody, shared preparation interfaces, observation lifetimes, native outcome ownership and unknown-authority behavior cross existing subsystem boundaries.

## 1. Objective, architecture and scope

Make Console Send immediately responsive and reduce repeated configuration, catalog, permission and native admission work. Each subsystem has one owner and explicit inputs, results, freshness, effects, failure and cleanup contracts. Future features should extend their relevant domain boundary without spreading new file reads through the UI and every provider.

Preserve enabled capabilities, workspace boundaries, hook consent/history, durable acceptance, required capture, cancellation, recovery, queue ordering and resource ownership. Unknown required authority deliberately refuses the affected operation instead of being interpreted as a successful off-switch observation. This hardening is a visible behavior change with focused qualification.

Acceptance targets remain rendered receipt/input feedback within 100 ms across ordinary, enabled-feature, approval, refused and error routes, and under one second of ordinary application overhead before actual provider-adapter entry. Cold initialization, approval waits, requested retrieval and provider/network time are reported separately; those routes still require immediate feedback. These targets are unmet until qualified evidence demonstrates them.

Choose a thin preparation coordinator with domain-owned I/O and immutable results, extending existing runtime/controller/store foundations. Per-service batching alone leaves duplicate consumers. Profile-wide storage actors/caches would add invalidation, scheduling and shutdown machinery without current evidence that they are necessary.

| Responsibility | Owner and contract |
| --- | --- |
| Request lifetime and issued native work | Existing app-owned ConsoleRuntime custody, extended to a bounded received intent and explicit promotion. |
| Preparation/run state, admission and queues | Existing controller/store and resident coordinators; retain their single authoritative state. |
| Preparation sequencing | Focused Chat collaborator with named dependencies and attempt-local sequencing data; no broad controller proxy or separate state ledger. |
| Selected configuration and provider requests | Existing turn context, preparation and prepared-request models. Complete context follows checked capture. |
| Tool data and policy projection | Existing MCP/source owners and providers, exposed through one explicit stock preparation result. |
| Hook consent and launch | Existing HookPermissions and hook runtime, with a named runtime decision-host review bridge. |
| Durable effects and recovery | Existing repositories, effect claims/completion and dispatch checkpoints. |
| Pixels, focus and status | Disposable views render projections and emit intents without owning surviving requests. |

The responsibility flow is Receive, Prepare, Commit, Dispatch, Settle. These are boundaries, not another state machine. Some preparation/effects require committed identities and remain after commit. Reuse the existing stock composition entry point while moving its orchestration into the focused owner. Preserve public/known callbacks and live lookup with explicit migration adapters.

### Atomic admission and promotion

The store/controller admission owner provides one authoritative per-session claim under its existing admission/preparation synchronization. Runtime custody owns task lifetime; it is not a competing admission gate. Every immediate entry point, including direct controller calls, queued/background origins and retry routes, consults that same claim contract. A queued entry uses its existing queue authorization, not a new manual reservation.

A received claim is an explicit opaque request/session identity and generation, not an incomplete ConsoleTurnPreparation. Promote only the matching claim to a complete preparation/request without releasing and reacquiring the session slot. Stop/Close seals, source validation and the promotion winner are checked in that same atomic boundary. Failed scheduling, capture or promotion releases only its matching generation; late cleanup cannot release a successor. Extend archive/inflight accounting and confirmed Close/quit projections to include received work as preparing, without calling it durably accepted.

Constructor/module setup performs no eager native loading, unrelated housekeeping or heavy provider initialization. Retain boot/module guards. Stable app-owned services and named live accessors have explicit binding rules. No screen callback or widget survives request handoff.

This specification develops the broader restructuring previously reserved by the [incremental Send design](2026-10-05-incremental-send-speed-design.md). Accepted small changes remain governed by their existing contracts. The uncommitted MCP-member candidate is reassessed on its evidence rather than automatically included.

The design was grounded in the integration worktree at d382e1cda1ede77a49da1c64391d2a36c53d6c4d, including its working-tree source. Earlier timings qualify only their recorded bases. Check current relevant source/ADR drift before implementation planning without waiting for unrelated work to finish.


## 2. Shared results, freshness and I/O ownership

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

Source dependencies are demand-driven within the domain operation. An empty MCP maximum does not disable local or builtin consumers; read the common switch if a participating consumer requires it, but skip an unused external catalog. A disabled family must not initialize its provider or perform its family-specific reads merely because another family needs shared policy. Preserve hook-history reconciliation even when no hook currently requests authority. Live and disposable-preview calls are separate operations: reuse within each, preserving inspector publication and explicit preview behavior, rather than sharing a result across them.

User-approved OPT-88 refinement (2026-10-09): a stock composition whose exact frozen tool-ID ceiling permits only builtin tools skips the external catalog, including its admission/migration/errors and audits of excluded external definitions. This is an explicit exception to preserving incidental catalog effects above, not a cached successful external observation. Unknown, mixed and external-capable ceilings retain fresh external reads and effects; the initial maximum, participating permission/switch reads and actual invocation gates remain fresh. The ordinary stock fallback uses the same decision, while custom routes preserve their contract.

The initial frozen definition maximum and the later live composition have different jobs. Keep the former as a narrowing ceiling; a later observation can remove eligible tools or refuse the result, but cannot silently expand the submitted maximum. Do not promise one permission read for an entire Send: admission, resumed approval and actual execution are distinct boundaries. Consolidate repeated reads within a boundary; retain reads required by an intervening change or effect.

A source/session replacement or relevant input revision change rejects affected publication. A Settings change must not silently substitute a new provider or workspace into a frozen request. Approval resume revalidates the relevant inputs and authority, using the existing review/retry outcome rather than granting clearance from the old result. Reuse unrelated pure results only if their own dependencies remain valid. No unbounded automatic rebuild loop, file-watcher authority, TTL permission cache or profile-wide generation that invalidates all results for every small change.

### Attempt ceilings and restart recovery

The no-widening ceiling applies to the lifetime of the attempt whose maximum was captured. An in-memory continuation retains its captured maximum under the existing retry contract. Existing durable checkpoints persist Library authority and destination, not the original MCP tool/definition maximum. Restart recovery therefore cannot pretend to compare against an unavailable old ceiling: it follows the existing fresh-context recovery, destination/Library compatibility and live execution-gate rules. Preserving an original MCP ceiling across restart would require an explicit persistence decision outside this no-format-change design. No receipt-only request is replayed automatically after restart.

### Native operation shape

The coordinator requests complete domain operations with declared inputs/results. It does not open files or manage transactions itself. A finite worker performs the needed reads and required related writes under its domain's existing admission and locking rules, then returns detached data after its native resources retire.

Batch declared related file targets and avoid helper-owned rediscovery where an equivalent same-operation contract can be established. Preserve per-target path/identity/pause checks and exact canonical source selection. Existing producer recovery observations remain independent where their authority contract requires them. A common profile does not make catalog, permission, hook and SQLite owners interchangeable.

Preserve lock order and existing thread/loop affinity. Pure inventory callbacks that belong on the app loop stay there; arbitrary callbacks do not run under storage locks. Each native worker segment contains no await. One semantic preparation API may use multiple loop/worker segments where the existing callback affinity or effect order requires them; fewer helper-owned reads does not imply one thread or one giant lock for every route. Cancellation retains custody until the issued native work actually retires, and then rejects obsolete publication. File-store changes and SQLite commits remain separate owned effects with defined failure behavior, not a claimed global atomic commit.

### Execution and extension

Tool invocation, hook launch, credential resolution, workspace access and durable mutations retain the live gates owned by those services. Builders consume preparation data through an explicit API; they do not obtain authority from an internal supplied-data flag. Public/custom adapters keep their documented late lookup and call shape until explicitly migrated to the new contract. Shared stock data must not bypass a custom receiver or reader; unsupported adapters retain their ordinary route. Validation of detached data is not a supplied-data authority bypass.

A future tool category adds its domain capture/result and pure catalog construction to the existing tool preparation boundary. It supplies its live invocation gate and required effect rules. It does not teach the UI, controller, history service and every provider builder to load the new policy file independently. The implementation plan must include a small concrete extension example and a measured I/O count to prove this boundary works.



## 3. Receipt, approval, cancellation, retry and errors

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

Domain acceptance notification and UI draft clearing have different contracts. Notify the domain according to the existing origin-specific acceptance rule, while the current view may clear text/undo only when the accepted request's captured input revision and view attachment generation still match at the actual clear action. Equal text is not enough: retyping identical text is a newer edit. Preserve known no-argument callback signatures through the UI adapter; do not pass new parameters blindly or suppress required domain effects merely to prevent a stale clear. Runtime work never holds the old widget callback.

New received-intent routes validate exact input/staging revisions before promotion. Existing complete-request APIs retain their documented attachment-prefix transfer behavior: only the requested prefix transfers, and a later suffix remains staged. Preserve object identity, order and exact restore/unwind without duplicating attachments. The implementation plan must distinguish these routes rather than impose the UI receipt's strict change rule on all callers.

The coordinator must not retain widgets or screen-bound callbacks after handoff. A lightweight intent includes detached selected values, stable identities/revisions and staged-input references only. App-owned providers finish checked configuration capture. View attachment/generation belongs to presentation, while request/session identity belongs to the surviving job. Manual button/Enter, spoken Send, queued turns, agent starts, wakes and dispatch recovery join the appropriate existing domain boundary without inventing a fake UI receipt for headless work. Slash commands and raw-console routes retain their own dispatch/refusal semantics.

### Unknown authority and error policy

At an authority-dependent operation, represent a failed required observation explicitly as unavailable/unknown. It is never coerced to false, an empty policy, permission disabled or an old successful value. Refuse the relevant operation before its effect, even when a previous approval exists. This intentionally hardens the current invocation helper paths that can treat failed kill-switch reads as off. It does not make every optional feature or diagnostic failure fatal to an ordinary chat. Apply the rule through the complete stock call chain, including the controller-injected local switch closure, builtin/MCP consumers and approved-call paths: a leaf helper cannot recover an Unknown value already converted to false upstream. Keep unsupported adapter behavior explicit until migrated. Known missing/default state remains distinct from a failed required observation.

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

### Durability terminology and the minimal dispatch barrier

The user reaffirmed the latency/ownership focus and selected the existing saved-turn failure policy: a failed acceptance save stops Send and keeps the draft; existing temporary chats remain available. Do not add an unsaved fallback, new persistence mode, stronger synchronization setting or storage redesign to this effort.

For the normal chat database, committed acceptance means a completed atomic SQLite transaction under the existing WAL/synchronous=NORMAL policy. It supports application-crash recovery; it is not a guarantee that the latest transaction survives operating-system failure or power loss. Automatic-work execution fences and native file owners retain their own stronger policies. No PRAGMA or file durability primitive changes as part of this clarification.

The saved-turn barrier covers the canonical user/assistant identities, accepted checkpoint, required destination/Library compatibility and transaction contributions needed to reconstruct or authorize the request. Required dispatch-start/trace admission, consent reconciliation and hook/provider-message handoff also remain ordered before their dependent effect. UI repaint and best-effort audit/index projections are not permission or acceptance authorities. Workspace projection is not presumed optional: current bindings may depend on it.

Prompt-history persistence is auxiliary to conversation recovery and remains an evidence-led optimization candidate, but its current awaited ordering is retained until a concrete owned queue/drain/failure contract is reviewed. This prevents an untracked background write from replacing a measured delay with a shutdown bug. Consolidation targets duplicate reads, admissions, worker handoffs and serialization inside these existing boundaries before altering the boundary itself.

Always measure acceptance commit separately from preparation and postcommit work. The earlier instrumented 0.527-0.812 s commit samples do not account for the much longer subsequent intervals and do not justify attributing six-second module delays to fsync. Current saved-turn acceptance already avoids forcing a WAL sync on every ordinary commit.

### Retry and effect order

Precommit Retry creates a new attempt and re-enters the necessary captures. Existing held-preparation actions determine which selected inputs are preserved or deliberately recaptured; retry does not automatically replay old hook side effects, consume another staged input or widen the frozen maximum.

Postcommit Retry uses the existing assistant/user identities and checkpoint. Re-enter `retry_dispatch_recovery`, including destination and frozen-authority compatibility, fresh provider resolution and the existing context/retrieval rebuild. The coordinator cannot turn this into a new Send. A provider-started/uncertain attempt retains the current conservative recovery path; no automatic network replay or global exactly-once guarantee is introduced.

Initially retain this existing required postcommit sequence: identity/settings and roleplay reconciliation; durable message-owner publication; exact staged-input clearing; workspace projection; queue acknowledgement; accepted hook/context handoff and manual accepted callback; ordered prompt-history append; preparation publication; fresh hook admission; existing trace/dispatch/provider boundaries. The label accepted_hook includes provider-message handoff, so it is not merely a disposable UI notification. Best-effort return/failure behavior remains owned by each callback. History remains awaited in this design; moving it after provider entry would require a separate ordering/recovery decision, not an untracked background task.

Use existing effect claims/completion and checkpoint transitions for repeated resumes. Record completion only when the effect's existing success contract is met. A coordination refactor must not wrap a whole sequence as one opaque completed effect or claim atomicity across independent stores.

### Native ownership refinement found during this design

Current `AgentChatStart` commit code retains and shields an explicit commit worker; ordinary manual commit can call the naked `_run_durable_db_call` to-thread await. The architecture must require explicit custody of every issued native future, including the ordinary commit path, with registration before await and physical outcome settlement before recovery/creator close. This is a lifecycle correction to verify, not a claim that existing task cancellation proves native rollback. Reuse the actual runtime/callback custody owners and per-attempt references; do not add a general task framework.

Current MCP `mutation_fence` is a process-local path RLock. It cannot establish cross-process write serialization, and an atomic file replacement is not by itself an atomic compare-and-set. The preparation refactor retains actual store mutation contracts; any stronger cross-process writer contract needs a separately explicit storage-owner decision and qualification. Cross-process revocation observations are still required at their existing live gates. This limitation must not be obscured by a generic snapshot revision or a claim of a transaction over several JSON files.

## 4. Extension, migration and verification

### Extension contract and concrete example

A new read-only workspace tool family registers its supported tool IDs and builders through the existing tool catalog provider seam. It consumes this attempt's selected roots, policy projection and immutable definitions from tool preparation. The family implements its own live path/exclusion/access gate and resource ownership. If it needs another source, its domain capture declares that source, freshness scope, required effects and error outcome. It does not add a settings read to every provider, change prompt-history ownership or acquire an unrelated hook lock. The builder is independently testable with detached data; actual execution is independently testable against the real workspace authority.

Dependencies remain explicit and named. Stock prepared-data APIs do not bypass custom readers or callbacks. Preserve known public signatures and late lookups with narrow adapters during migration; migrate supported adapters deliberately. Normal private relocation uses ADR-220's existing compatibility limit. No reflected dependency bag, generic step graph, new scheduler, cache TTL, duplicated state ledger or configurable public legacy/fast mode is needed.

### Migration boundaries

The enablement prerequisites for early runtime receipt are unified atomic admission/promotion, explicit issued-native-work custody, screen-free capture and the runtime-owned initial hook-review bridge. Implement/qualify those contracts behind the current entry points before activating earlier custody; the bridge cannot wait for a later hook-batching slice. The first enabled receipt slice covers both ready and approval-required paths without retaining a screen dispatch closure. Each slice has its own rollback and qualification boundary.

1. **Receipt and custody:** establish the enablement prerequisites above, then activate bounded intent/promotion and screen-free capture behind the current entry points. Preserve draft/attachment/queue semantics and prove real input responsiveness. Keep final domain authorities and durable ordering unchanged.
2. **Tool domain:** add one explicit preparation API/result used by the stock MCP and local composition consumers, with nested immutability, covered observations and required mutation/audit outcomes. Preserve the initial maximum, live invocation gates and custom fallback contracts. Migrate both live and disposable preview consumers with their existing publication distinctions.
3. **Hook domain consolidation:** reuse the actual consent/reconciliation owner and the runtime decision bridge already qualified before receipt enablement. Consolidate same-boundary reads without removing consent-history transitions, final launch checks or event-specific error rules.
4. **Lifecycle/effects:** unify ownership of issued native work, including ordinary commit; retain current checkpoints and effect order. Move only the preparation orchestration into its focused owner, removing each migrated duplicate path rather than retaining two long-lived pipelines. Keep history awaited.
5. **Qualification and cleanup:** prove the final integrated source, remove temporary adapters that no supported caller still needs, record remaining bottlenecks, and update task/ADR/docs. Unproven micro-optimizations are not a prerequisite; reassess the earlier uncommitted MCP-member candidate under the new contract rather than automatically carrying it forward.

These boundaries can be separate atomic implementation tasks after written-spec and plan approval. Each migration must cover the relevant manual, queued, background and retry consumers rather than qualifying a fast path while leaving alternate entry points unsafe. The design task references only existing tasks; future implementation tasks will be created in dependency order after the design is reviewed.

### Evidence and performance requirements

Reuse existing focused preparation, runtime lifetime/shutdown, hook review, queue, dispatch recovery, MCP read-error/snapshot, native storage and cancellation controls. Add checks for the new intent/promotion, unknown-authority refusal before effects, single-consumer continuation, deep immutability, narrowed maxima, callback compatibility and the explicit native commit future. Important interleavings include duplicate Send, a new draft while approval is open, source/profile/global-switch changes, a raced downgrade write, Stop during commit, repeated cancellation, Close during publication, navigation/remount, and late worker success/error. Preserve original assertions, physical ownership evidence and existing deadlines; fixtures or direct action calls alone do not qualify the product.

Count original file loads, native admissions, path checks, writes and worker handoffs by domain and operation. Verify that a new pure consumer of the shared result adds zero native reads/admissions. Bound nested-result size and avoid repeated freeze/copy work. Distinguish necessary new observations after an authority boundary from repeated helper work within the same operation. Do not add global monitoring or permanent detailed diagnostics to the production hot path.

For final latency, drive actual Enter and button events through the real app pumps (including eager scheduling) and observe rendered Preparing/approval/refusal/error frames and input responsiveness while a real native operation is held. Record event-to-rendered-receipt and event-to-actual-adapter entry with one monotonic clock. Require receipt/input feedback within 100 ms and ordinary application overhead under one second; separately report cold initialization, enabled features, deliberate approval waits, retrieval and provider/network time. The time exclusions do not exempt those routes from immediate feedback. If the first slice misses the adapter target, report the gap and attribute the remaining work rather than redefine completion.

Always report unadjusted event-to-render and Send-to-adapter elapsed samples first. For the ordinary acceptance route, application work includes configuration, policy/catalog reads, native admission/path checks, worker queuing, serialization, commits, history and trace setup. A phase name such as initialization or preparation cannot exempt work triggered by Send. A cold sample retains its full elapsed cost and is labelled separately; it cannot be removed from an overall performance claim. Only actual bounded human-wait, requested retrieval/provider work spans may be separately attributed, with their scope recorded. The ordinary enabled-tools stock route must be measured alongside an empty/disabled route; optional provider/plugin routes retain explicit capability and fallback observations. Any subtraction is shown alongside the raw samples, never substituted for them.

Run matching private baseline/candidate comparisons with the same source qualification, interpreter/dependencies, filesystem placement, feature shape and minimal instrumentation. Alternate run order and retain individual cold/warm samples. Require real persisted replies, complete trace links, correct checkpoint settlement and positive native/process retirement; forced termination or source drift invalidates a performance claim. Qualify Windows, Linux and macOS separately before claiming supported-host acceptance. Run targeted tests only; the full suite remains opt-in.

## 5. Governance, acceptance and handoff

ADR-225 extends ADR-094/098/126/148/163/197/220. It records received-intent custody/promotion, finite domain results, observation versus live authority, intentional unknown-authority hardening, native outcome ownership and compatibility limits. Existing storage formats, native qualification, durable effect order and recovery semantics remain governed by their owners. Stronger cross-process writer coordination or reordered history needs a separate explicit contract behind the same domain APIs.

Render existing Preparing, approval-needed, stopping and refusal/error projections with the governing design tokens and component patterns. Preserve focus, composer undo, queue/Stop behavior and generation-fenced publication. Add no keybinding, screen, public fast/legacy mode or implementation terminology to the product. Bound synchronous input/result normalization so reduced I/O does not become repeated large copies on the app loop.

Retain per-session ordering, current queue/global caps and bounded native concurrency. One slow job must not hold a profile-wide execution lock. Several requests affected by the same hook-policy change still revalidate and consume their own continuations once through the resident decision host.

Result size and field validation follow existing domain/input limits; oversize or malformed data uses the documented refusal/fallback without silent truncation. Provider-specific serialization may produce its own wire representation without mutating shared data. Diagnostics retain bounded stage/count/timing facts rather than secrets, prompt bodies or automatic project-instruction bodies.

Written-spec review precedes writing-plans, and plan review precedes product implementation. Execution remains inline without subagents under the user's existing instruction. The design task closes after the completed written review and requested audit; product implementation remains gated on plan review. No full sweep is authorized, and no product improvement or runtime regression acceptance is claimed by this documentation deliverable.


## 6. External architecture references (non-normative)

Reviewed on 2026-10-06 following reference research delivered from the user's side conversation. The sources below were independently checked for the listed observations. Repository links use mutable main branches, not immutable commit pins; recheck any implementation detail before relying on it. Claude Code observations come from public documentation, not its private engine internals. These references support the chosen architecture but establish no tldw latency, I/O-count or retirement acceptance.

### Codex: routing acknowledgment and retained approvals

SessionIo.submit_turn_input queues an identified turn-input operation and waits for Core's routing decision. Dropping that waiter does not retract the queued operation. Command approval registers its responder in active-turn state before emitting the request, keys responses by call/approval identifiers, and treats a cleared pending approval as an abort. [Public session source](https://github.com/openai/codex/blob/main/codex-rs/core/src/session/mod.rs).

The app-server API separates turn start/steer/interrupt requests from streamed turn/item notifications. [App-server documentation](https://learn.chatgpt.com/docs/app-server).

Application to this spec: receipt, routing and completion remain distinct; runtime-owned request/approval identity survives presentation changes, and cancellation is an explicit operation. The protocol does not prove that a terminal frame paints within our budget.

### Oh-my-pi: preparation cancellation and complete-path timing

AgentSession.prompt enters admitted-submission accounting and records submission time before asynchronous preprocessing. Prompt setup retains its generation and AbortController. Abort invalidates setup, cancels preparation/post-prompt work and waits for agent idle; a dropped-prompt callback restores text that never reached the agent/session file. [Agent session source](https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/src/session/agent-session.ts).

Application to this spec: cancellation covers capture and preparation, exact undelivered input remains recoverable, and raw timing begins at Send rather than provider entry. An agent idle signal is not a substitute for tldw's native physical-retirement evidence.

Its TUI coalesces ordinary render requests and uses completed frame cost/output backlog to schedule repaint work. [TUI source](https://github.com/can1357/oh-my-pi/blob/main/packages/tui/src/tui.ts).

Application to this spec: use Textual's existing coalesced refresh mechanisms for projections rather than introducing another renderer. Preserve meaningful receipt/approval/Stop/terminal publication; render coalescing must not discard transcript data or clear unseen receipts without their existing paint evidence.

Durability differs: ordinary local appends are synchronous after a lazy creation boundary but do not fsync. A new ordinary session can remain memory-only until an assistant message or explicit disk creation; remote indexed publication uses ordered asynchronous queues with explicit flush/drain. [Persistence guarantees](https://github.com/can1357/oh-my-pi/blob/main/docs/session.md#persistence-guarantees-and-failure-model).

Application to this spec: retain tldw's own durable-before-dispatch and recovery boundaries. These alternative persistence choices cannot justify moving required commits out of our critical path.

### Claude Code: operation-boundary policy and deferred definitions

The public Agent SDK permission flow evaluates hooks before deny/ask rules; a hook allow does not bypass those later constraints. Its approval callback handles calls unresolved by earlier stages, so it is not itself a universal per-call gate. [Permission evaluation](https://code.claude.com/docs/en/agent-sdk/permissions).

Application to this spec: retain hard floors at their actual execution boundaries, including previously approved routes; never infer authority from a prepared result or optional callback alone.

Official documentation describes MCP definitions and skill contents loading on demand. [How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works).

Application to this spec: domain preparation is demand-driven. Definition/context loading is separate from current consent, storage admission and execution checks; the documentation does not establish that those checks are eliminated.

### Adoption boundary

Use these references as examples of explicit submission outcomes, backend-owned approvals, preparation cancellation, demand-driven data and bounded rendering. Adapt the ideas through existing tldw owners and APIs. Adopt no external engine, weaker persistence policy, permission shortcut or numerical performance claim. The reviewed spec and ADR-225 remain the normative contracts.

### 2026-10-09 stock composition refinement

ADR-225's OPT98 amendment supersedes receipt-time MCP maxima for qualified stock
owned capture: explicit demand defers definition/eligibility observation to each
existing execution consumer's fresh composition. The provider freezes its issued
IDs/hashes; prospective hooks, live runs and disposable previews retain separate
lifetimes. Explicit captured/custom maxima, plugin narrowing, durable acceptance
and live invocation authority keep their contracts. See the canonical ADR and
[composition ceiling plan](../plans/2026-10-09-console-composition-tool-ceiling.md)
for error timing, compatibility and qualification; no cross-attempt cache is added.
