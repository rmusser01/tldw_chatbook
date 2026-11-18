# ADR-230: Console Send preparation and I/O ownership

Date: 2026-10-06
Status: Accepted following written review and the requested issue audit.
Task: [TASK-34563](../tasks/task-34563%20-%20Design-Console-Send-preparation-and-I-O-ownership.md)
Spec: [Console Send preparation architecture](../../Docs/superpowers/specs/2026-10-06-console-send-preparation-architecture-design.md)
Extends: ADR-336, ADR-098, ADR-126, ADR-345, ADR-163, ADR-197 and ADR-220.

## Context

The enabled Console Send path repeatedly observes configuration, catalog and permission sources through separate helpers and native operations. Measurements show substantial application delay, but overlap and instrumentation prevent additive savings claims. Moving existing bodies alone would leave repeated work and unclear lifetimes.

The Console already has app-owned custody, authoritative preparation/run/queue state, immutable request models and durable dispatch recovery. UI capture nevertheless precedes custody, and native commit retention differs between ordinary Send and AgentChatStart. Shared preparation needs explicit source, freshness and effect boundaries rather than a general cache or another job system.

## Decision

1. Extend existing runtime custody and per-session admission with a bounded received intent. It grants no execution authority or durable acceptance. Publish Preparing feedback, perform screen-free checked capture, then promote into the complete existing request. Preserve exact input revisions and staged-input transfer/unwind.
2. Use a thin Chat preparation coordinator with named dependencies. Existing runtime, controller/store, decision, queue and repository owners remain authoritative. Responsibility phases preserve dependency order; required work may remain after durable commit.
3. Produce domain-specific immutable results for one operation/attempt. Stock MCP/local builders share normalized catalog/policy and compose-time switch data. Pure consumers perform no repeated native freshness scan. Submitted maxima are narrowing ceilings; observation metadata is not permission authority.
4. Keep I/O, locks, native path/source/pause checks and mutations inside finite domain operations. Batch declared related work under an equivalent contract. No native lease crosses an await or approval wait. Independent stores retain separate transactions; process-local fences do not establish cross-process serialization.
5. Preserve hook reconciliation, token/grant retirement, final launch serialization, source/revision conflicts and event-specific errors. Present initial consent review through a named runtime decision-host bridge without retaining a screen continuation.
6. Retain exact issued native futures through repeated cancellation and settle their actual outcomes before recovery/retry or creator close. Apply this to ordinary commit as well as agent-start work. Preserve postcommit order, effect completion and checkpoints. History stays awaited; uncertain provider calls are not automatically replayed.
7. Unknown required authority refuses the affected operation before its effect, including prior-approved calls. This deliberately hardens stock switch-reader errors that can report off after failed reads. Missing/default/corrupt/uncertain observations remain distinct. Optional fallback and best-effort audit/display policies remain separate.
8. Preserve supported signatures, known live callbacks and custom routes with explicit migration adapters, using ADR-220's private-relocation compatibility limit. Features extend domain capture/result and execution-gate contracts without a generic step graph, dependency bag, scheduler or permission cache.

## Explicit refinements and preserved boundaries

ADR-336's app-owned accepted-turn contract is extended earlier to received-intent custody. Complete request construction and durable acceptance remain separate. Navigation still detaches presentation; initial review can now remain app-owned.

ADR-126 retains exact storage membership, recovery, physical ownership and retirement. This decision requires explicit ordinary native-work outcome retention; it creates no transferable lease, new file format or stronger cross-process transaction.

ADR-197 retains observed consent history and launch/revocation requirements. Initial review gains a runtime bridge. Refusing unknown authority does not change hook execution-error fail-open rules after authority was established.

The stock tool error refinement must cover actual switch/permission and approved-call entry points with focused tests. It is not claimed as an existing universal deny-on-uncertainty property. Stronger store writer serialization or reordered history requires a separate decision behind the same domain boundary.

## Written-spec review clarifications

Received reservation and complete promotion use the same atomic admission owner across entry points, with exact-generation cleanup and no release/reacquire gap. Runtime custody remains lifetime ownership. Safe native custody and the runtime hook-review bridge are both prerequisites to enabling early receipt.

Tool ceilings are attempt-bound. The existing durable checkpoint does not persist the original MCP maximum; restart follows its existing fresh-context and Library/destination compatibility rules. This decision adds no hidden maximum persistence or restart replay.

Domain acceptance and generation-checked UI clearing remain distinct, including retyped identical text and attachment-prefix compatibility. Unknown-authority hardening covers upstream stock closures as well as leaf readers. Domain captures are demand-driven, and live/preview operations retain separate publication lifetimes.

Performance qualification reports raw elapsed samples and includes Send-triggered application setup. Phase labels cannot remove that cost from an acceptance claim.

## Alternatives considered

| Alternative | Tradeoff |
| --- | --- |
| Per-helper shortcuts/batching alone | Small changes leave duplicate consumers and fragmented ownership; retain only independently useful changes. |
| One profile-wide worker/cache/lock | Adds invalidation and shutdown rules and can serialize unrelated chats; current evidence does not justify it. |
| Generic pipeline engine or duplicate state ledger | Obscures authority/effect order and duplicates resident owners. |
| Long-held native snapshot through approval/dispatch | Violates finite ownership and live revocation/recovery boundaries. |
| Move history/audits after provider entry | Changes ordering and shutdown; history remains awaited and audit policy stays owner-defined. |

## Consequences and qualification

Cross-module contracts and received-intent promotion require phased migration and source-current qualification. No schema migration, dependency, keybinding or public fast/legacy mode is required. Unknown-authority hardening can refuse calls that legacy helpers might continue; this is intentional and separately verified.

Acceptance requires actual rendered/input feedback within 100 ms and ordinary application overhead under one second to adapter entry, unchanged required durability, and positive native retirement. Observe original I/O counts and minimally instrumented cold/warm comparisons. Qualify supported hosts separately; full sweeps remain opt-in. Moved code, a faster empty path or a successful reply does not establish acceptance.

The linked spec defines lifecycle, freshness, errors, compatibility and verification. Written review precedes implementation planning and plan review precedes product changes. TASK-34563 records the completed design review; product work remains gated on implementation-plan review.


## Durability terminology clarification

The user selected the existing saved-turn failure policy: failed acceptance keeps the draft and refuses dispatch; existing temporary chats remain available. The current normal chat commit uses SQLite WAL/synchronous=NORMAL, providing application-crash recovery without promising power-loss survival of the latest commit. Preserve separate execution-fence/native-file policies. No new persistence mode, unsaved fallback, PRAGMA change or storage redesign is introduced.

Keep the minimum canonical acceptance/checkpoint and required consent/trace/context facts before their dependent dispatch/effect. Classify auxiliary projections without treating every postcommit callback as optional; history remains awaited pending an explicit owned deferral contract. The implementation focus stays duplicate native reads/admission, fragmented ownership, serial handoffs and real input/render responsiveness.

## Renumbering and current integration

Original source decision at c504ff3b1bda is preserved above with its original approval/Proposed status. The older current-dev ADR222 HTTP-session and ADR223 embedding-cache decisions keep their identities; younger Send-preparation222 and polling223 become230 and231 after the all-ref/worktree census. Existing34406/34563 child task IDs and historical qualification remain. The general polling proposal stays Proposed; only its explicitly accepted sole-stock-manual-Preparing amendment is selected. Current integration composes resident acknowledgement/captured-draft/custody/human-decision/Goal/Personal Context/response-rule owners, not a parallel ledger, cache or scheduler. Actual current native, original callback, final saved-source and rendered-input/transport evidence must qualify the original bounds before completion; old source counts or server-frame emission do not establish that proof.


## Current-owner integration amendment (2026-10-13)

Current integration retains Console automatic-submission, Goal/Personal Context, Rules helper pool and machine-followup custody owners. Received manual intent grants no new automatic allowance. Keep shared history/native-hook budget and deadline accounting, exact origin/capability, generation checks across awaits, immediate destructive Rule revocation before writer waits, and physical preparation/transport retirement before creator close. Settled Goal automatic repair remains refused.

### Current mounted authority reconciliation — 2026-10-13

The current mounted adapter already owns a checked `turn_admission` pre-capture. Preserve that owner and its matching/fallback decision. The shared snapshot producer may receive its already captured change-review roots, character authority, prompt transforms and skill maximum as explicit data inputs; it must not repeat those native reads or retain the UI owner. All other selected/session fields continue through the same producer. These inputs add no permission or durable acceptance: current live invocation/dispatch checks remain authoritative. Unknown/custom callback affinity and current cache compatibility remain with their original adapters. ADR required: this amends existing230, no new ADR. Native parity and no-repeat controls precede mounted adoption; standalone result parity is not full Send acceptance.


The complete legacy UI custody route may attach its original synchronous draft revision to the existing request/claim. Its successful same-stack captured-prefix commit may record one exact resulting revision on that claim, only while the original captured revision and session slot still match. Promotion retains this explicit owned-transfer witness for cancellation restoration; it never snapshots a later draft. Later authored empty or identical text has a distinct revision and survives. Default detached/custom requests keep no witness and their existing preparation identity/API; explicit recovery-shelf restoration retains its existing user-action contract.


Current shared-tool preparation retains TASK33620.15.1 catalog reuse through its existing owner. The qualified detached result carries the exact server/connection observations from its catalog records (or an explicit incomplete-inventory None), never inferred after adoption. Existing fresh fingerprint, manifest/governance, file-stamp/settlement, admission-epoch, composition-key and connection checks remain required before retention/reuse. A single permission payload decode may serve MCP/local composition, while fingerprint native reads remain separate real work. No new cache, execution authority or public custom-provider arguments.

### 2026-10-14 — Connected Skill defining ownership in current integration

Place the existing connected Skill construction and finite trust-preparation
bodies in a real SkillServiceWiringMixin and actual app_skill_wiring functions.
ServiceWiringMixin inherits the descriptors normally; its existing public
properties and original importing/patch slots remain. Captured bound callbacks,
live Python defaults, original coroutine bodies and App/Service/instance custom
shadows keep their existing dispatch contract. No forwarding alias, additional
service or general provenance/custody framework is introduced.

Moved bodies continue reading the live original app_service_wiring collaborator
slots. The existing consumed source records address the actual defining class
and module. At finite capture, retain the original module and validate its
already recorded helper pointer before retained proof/loop/thread reads; keep
that original module through the issued worker's existing await/error/drain
path. This is ordering of existing source refusal, not a new blanket body or
default policy. Separate published-plugin/builtin checks and unconsumed custom
construction routes stay separate. Physical retirement and original permissions
remain required before publication/creator teardown.

Exact scoped proposal4be7147c and independent reviewf17aa6ea bind the connected
35-owner relocation and six explicit retained-pointer corrections. Causal stock
Skill/Collections native-held negatives precede the3 passing prospective
controls. This releases only the reviewed current-owner integration and its
affected qualification; native optimization, whole-Send timing, final-base
count pins and full acceptance remain open.

### Current run-log path preparation

TASK34563.6 shares one sensitive-path context only within the exact current ordinary RunLogWriter bind operation. Preserve its two independent containment checks, original migration and effect ordering, and the original captured containment callback. Qualify the already consumed writer, predicate and resolver records before each check. A changed or customized callback, method body or default uses the existing two-argument path and fresh resolution; sharing grants no authority and is not an atomic snapshot. Context preparation intentionally precedes a possible path refusal, matching the selected original task. No cross-bind cache, source capsule or new storage policy is introduced.

The reviewed three-file proposal and original ordered 116-case regression are recorded in the PR3023 evidence directory. Six held-source controls additionally exercise the actual writer's native lease and hold, original denial/error boundary, custom migration arguments and physical retirement. Saved real-path qualification remains required after integration.

### Checked standalone cost derivation — 2026-10-14

TASK34406 AC5/51 routes the existing zero-argument cost builder through the existing checked synchronous presentation owner. Select the current session and store once in the original order; ownerless state clears before configuration/pricing work. The original derivation body and optional error boundary remain. A before-body or post-body refusal retains the last accepted display, cache classification and two outer pricing memo references, so refused work cannot change the real TTL timer or seed a later memo hit. The existing coalesced refresh selects a fresh pair on retry. Custom zero-argument builders and the original ten-second timer keep their dispatch contract.

Original direct-entry and actual armed-TTL controls reproduced blocked loop progress under held configuration locks. Candidate d0d88f61 and independently reviewed original/new controls require genuine installed-path qualification with original bounds, real post-check refusal, ownerless Timer.stop/task join and fresh-owner/source replay. This extends the existing checked display contract; it adds no authority, general rollback or scheduler. Full branch/current-base/native/latency acceptance remains open.

### Exact owned descriptor retirement — current Task17 integration

Known descriptor cleanup uses only the existing exact issued raw operation, active process/thread/task ownership and registered FD membership. It does not re-admit new work through source/parent checks after revocation. Voice/visual precedence and the preceding checked fallback for unrelated, copied, foreign or inactive ownership remain. Uncertain known custody still reaches the original close owner, which refuses native retry and retains resources/leases. Subsequent open/read/write/publication gates remain fresh. The current source review and original ownership/count/uncertainty plus affected lock/posture controls qualify this bounded seam; no general native retirement, platform or whole-Send timing claim follows.

### Current draft-cost settings freshness

Current cost R3 correction before qualification: independently reviewed source77069a2f/a3fc4cd2 fixes two reached draft-estimate regressions in R2 while preserving checked warm behavior, original getter/error/ownerless/zero-arg ABI and three-reference refusal rollback. Only an existing same complete stock owner/source/current-Runtime projection with settings-revision-only drift selects the existing fresh nonblocking native config owner. No fallback follows presentationFalse; actual native post-body full-key/closing/Runtime/disposed checks retain prior display and request existing fresh retry. Qualify exact original two draft regressions, original four causal/ownerless/checked refusal cases and seven real held-settings/native postbody/lifecycle controls; all original bounds/native/Timer/error/source gates retained. Root source and TEST reviews precede guarded temporary installation only of a3fc, freshfa612 baseline and explicit current Task-governance rebind with all79AC unchanged. No task/source/nativeall/platform/base/latency/wholebranch completion. ADR required:no new ADR; existing126/230 clarified for this same-owner native selection. Original active-edit Mock failure reproduces on baseline and remains separate.


### Current release structural correction — 2026-10-15

Keep original Actor Pack recovery worker capture, fresh Context, active Worker, finite read set and shutdown fence/drain in the existing ServiceWiringMixin service owner. The boot table still captures the same bound callback at starter invocation. Keep received draft validation and handoff publication in the existing input mixin under its original public method name, removing only the redundant store forwarding body. Shared mixed-head FIFO payload/snapshot/projection bodies may reside beside initial hook review on the same resident interrupt instance; controller accessors, lock acquisitions and source-bound pending readers stay unchanged. These are faithful ownership moves, not a new service, registry, callback wrapper, source-proof shape, authority or cancellation policy. Existing inherited size failures remain recorded against exact current dev; this release removes its added growth without raising a cap or changing the measurement.

### TASK-34406.2 — Admit explicit run writers and saved history at use (2026-10-16)

A stock Console run whose actual injected stock writer has an explicit root
admits that root before binding, instead of resolving an unrelated global
history root. Initial config/profile, database, workspace and permission admission
and physical worker/resource lifetimes still apply. Exact writer class, original
path/bind callbacks and consumed root must qualify; implicit, custom, changed or
stale public writers retain the global entry route. No admission decision is cached.

Conversation-scope saved-run search selects its current global root while the
profile/workspace scope is held, then retains history admission through directory
probes and actual segment loading. The canonical resolver consumes that admitted
root explicitly; a missing root never triggers a second implicit selection.
Custom readers retain their preceding one-argument ABI under the same scope.
This deliberately moves refusal for an unrelated global history source to its
actual search; refusal for the live writer remains before bind/provider entry.
Prompt-history persistence, audit order and durable checkpoints are unchanged.

Rejected: simply skipping global entry admission without guarding older-history
loads, or admitting one root and then reselecting another. Qualify real native
refusal/retirement and matched first/later sends; a successful reply alone does
not establish the broad latency or platform targets. See TASK-34406.2 and the
companion ADR-126/ADR-230 amendment.

The explicit writer root may be passively qualified from already loaded stock
source objects before entry and included in the initial admission observation.
That single execution scope still freshly admits every actual source before the
run body starts; it avoids re-admitting the same profile/config/database sources
in a nested scope. Qualification never imports or calls an optional source before
admission. Missing or custom source objects keep the admitted global fallback.


### TASK-34406.4 — Speculative screen imports yield to Console custody (2026-10-16)

The app-owned whole-registry screen preimporter defers its next route while the
existing `ConsoleRuntime.has_custodied_turns()` projection reports received or
accepted work. This is only a scheduling hint read by the existing daemon
thread: it grants no permission, retains no native lease and changes no
preparation worker, freshness check, audit or provider contract. An import
already running finishes normally. Between-route waits remain shutdown-aware;
the navigation-only park retains its original bound. A long Console response
can delay warming unvisited screens until custody clears; direct navigation
still uses its ordinary loader. The certain-work, single-route splash warmup
remains immediate. Prefer this existing custody projection over a duplicate
busy flag, extra executor or permission cache.

### TASK-34406.11 — bounded interpreter handoffs during finite preparation

Existing finite synchronous preparation callbacks may lower Python's interpreter-wide thread handoff interval to at most1ms while physical callback bodies run. Live original-function profiles identify a scheduling convoy across many GIL-releasing metadata syscalls, rather than a single269ms permission-file operation. This changes scheduling, not observation: every original permission, source, generation, path, epoch, native acquisition, error and retirement check still executes at its original boundary.

Overlap is counted under the existing finite-read bookkeeping lock. The first physical callback captures the prior interval; the last restores it only if the interval is still the value this scope installed. An already shorter interval is preserved, as is a later different external setting. Callback exceptions and repeated caller cancellation cannot end scheduling ownership before the physical body exits. No native lease, result, deadline, timer or authority is cached or broadened; no new worker pool, scheduler service or public setting is added. The interval affects all interpreter threads during that bounded window, so exact unprofiled first-send comparisons must demonstrate a whole-path gain before release. Profile-only or syscall-count improvements are insufficient.


### Known changed Character display conversation — 2026-10-17

TASK-34406.16 may omit the preliminary Character display change-detection read only when the previous immutable scope fingerprint exists, its open-conversation identity already differs from the current identity, and the existing stock finite-refresh selector qualifies the original display closure, controller, file-backed database, source methods and owners. This observation proves that a refresh is needed; it grants no freshness, row or execution authority. Call the original refresh immediately, retaining its original pair/groups/pair operation, source and ambient checks, final fingerprint comparison, retries, cancellation and physical native retirement before publication.

Unchanged conversations, missing/invalidated fingerprints, explicit forced or direct actions, custom/memory/changed sources and unqualified closures retain their preceding preliminary scope route. The original one-second active recheck, two-second idle display memo and all native deadlines remain. No general polling narrowing, deferral of checks, new cache, transaction, lifecycle owner or authority policy. Causal controls must prove the remaining original native reads and refusal boundaries, exact native retirement, and a fresh retry after actual metadata change. Release still requires consistent unprofiled actual Enter-to-HTTP improvement and current-base CI/review.


## Accepted review-boundary metadata exception (TASK-35582, 2026-10-27)

Adopt the [ADR281 focused amendment](281-primary-agent-multi-model-code-review.md)
for a host-accepted exact review-boundary wake only. Immediately after existing
automatic acceptance and before capacity preparation, await the existing bounded
context-window operation with the actual pinned ConsoleProviderResolution. Reuse
current TTL/singleflight/egress/size limits and an outer deadline bounded by normal
provider validation and original automatic-chain time. No new global wait flag,
settings re-resolution, expired capacity or changed destination is authorized.

Accepted authority/source/grant/session/Stop/priority fences are checked around
await with no lock/native lease held across I/O. Plan the normal tools afterward;
a missing reachable next-stage catalog is a visible pre-dispatch blocker. A failed
accepted preparation keeps its consumed finite charge and reports; no automatic
replay/refund. Ordinary manual/progress/Goal/queued preparation remains nonblocking
and retains existing4096unknown bootstrap. Current shared probe cancellation/close
ownership remains; metadata is not a physical generation attempt. No schema change.


### Source correction to accepted-wake carrier diagnosis (Ruling63)

Exact BASE7952422 controller source deliberately projects automatic SYSTEM
transcript notices as payload-only trailing USER while preserving AGENT_WAKE
authority and order. The prior reconstructed SYSTEM-carrier diagnosis was not
the actual wire path. Its carrier conversion/Safe-capture clauses are superseded;
ordinary serialization and the strict engine stay unchanged. Accepted-review-only
pinned capacity/revalidation and normal schema reachability requirements remain.
The full handler-backed fixture instead exposes completed checkpoint forwarding
without native replay rows. A focused replay contract is pending before source
changes at that extra boundary. Original diagnostic evidence is retained with
QA/ac24/review-wake-repair-design/diagnosis-controller-correction.md. No authority,
schema or additional live allowance follows from this correction.


### PR3050 current-owner reconciliation — 2026-10-31

The remaining Send work is rebased onto current dev owners. A pressed resident
snapshot survives the acknowledgement paint and later draft edits; only that
snapshot can retire its original draft after durable acceptance. Guardian may
transform the submitted body while the original snapshot remains the cleanup
witness. View publication retains the pressing view's attachment generation.
Custom mounted capture uses the existing finite preparation-read owner until the
physical callback retires, including repeated cancellation. Existing admission,
Stop, source, Goal, response-rule, transferred-draft and citation fences remain.

Stock MCP execution consumers freeze their own fresh definition ceiling during
composition. A deferred capture refuses changed factories/catalog sources before
reading them; invocation permission is still read fresh. Builtin-only captured
maxima avoid external catalog work. Hook observations are reused only within the
same exact attempt and owner, with fresh final admission and dispatch checks.

Recovered-image tasks belong to the existing view-worker group. Unmount releases
runtime and settings ownership before image admission is sealed, completes view
cleanup, and joins actual reader retirement in finally. App shutdown also joins
already detached workers. Reader failure or caller cancellation cannot skip the
terminal/speech cleanup by preceding it.

Current dev's native descriptor/admission generations, operation-owned DB scopes,
character refresh and one-second configuration projection remain authoritative.
The earlier Windows watcher and broad polling optimization are historical
qualification material, not selected replacement implementations. No new cache,
storage policy, authority, or ADR identity is introduced. The original under-100ms
input and under-one-second overhead claims and complete platform/Close
qualification remain unproven; TASK34601 and TASK34601.1 stay In Progress.

### TASK-40608 — bounded Windows native handoffs (2026-11-13)

The first shared-display/finite-DB candidate still measured 109.2348/174.0021 seconds for the unchanged README/edit journeys. An isolated warm native admission took roughly 15 ms; volume qualification queries contributed only about 0.4 ms, so changing that security proof is not selected. A controlled competing-Python-thread probe measured two identical admissions at 17.2062 s with the default handoff interval, 18.0002 s at the existing 1 ms scope, and 0.2831 s at 100 microseconds (10 microseconds brought no further benefit).

The 100-microsecond trial was rejected after the unchanged full journeys regressed to 267.9695/347.4966 seconds (gate 2, all original assertions passed). The two-thread microbenchmark did not represent the application's concurrent workload. Restore the original finite preparation limit of 1 ms and the original DB callback scheduling. Retain the independently tested no-op interval assignment fix, which preserves an already-shorter setting exactly. Do not hold the GIL across blocking native IO or retain a short interval across whole provider/human waits.

The next selected change consolidates actual display DB lifetimes under existing owners; changing a scheduling constant is not a substitute for that batch. The unchanged original journeys and all three pairs still decide the five-second gate. See the task's sub-five plan, gate-2 receipts and external native-handoff-probe receipts.

### TASK-40608 — permission command process adapter (2026-11-14)

Extend the finite domain boundary with the [reviewed contained permission owner](../../Docs/superpowers/specs/2026-11-14-task-40608-isolated-permission-design.md), under ADR-126's corresponding amendment. Existing async producer/future custody retains the physical synchronous request through repeated cancellation. The parent keeps its actual source/config/callback receipts and creator-only adoption; child source objects, generation counters and native leases are not parent authority.

Preserve internal selected-source rendezvous, external parent mutation_fence reservations, separate audit/DB ordering and captured load's publish=False behavior. Changed observations invalidate obsolete parent publication tokens before child continuation; unchanged observations preserve identity and suppressed events create no token. Positive command retirement, including retained non-raising hold failures, gates success, caller stamping and pending-clear. A known persistence outcome may update the conservative mirror after awaiter cancellation, but cannot publish a normal result to a cancelled or stale UI owner.

This selects a concrete permission-specific adapter, not a generic process pool or profile-wide storage service. Initial implementation remains disabled for production routing until the complete stock persistence surface and maintenance lifecycle are qualified. Setup/spawn/config import, IPC/check rendezvous, result adoption and retirement remain measured application work. No new TTL, blanket authority cache, skipped approval check, scalar replay or stronger cross-process transaction is introduced. Original TASK-40608 timing and integrity gates remain open until measured.

### TASK-40608 — finite admission observation process (2026-11-14)

The [private observation-owner contract](../../Docs/superpowers/specs/2026-11-14-task-40608-admission-observation-owner-design.md) was evaluated with ADR126's corresponding amendment and rejected after Task38: all six measured warm public acquisitions regressed. Production routing was never enabled. Remove the experiment's hooks and modules using exact preimages, retaining its replayable diagnostic evidence. The following describes that historical contract. The actual app/profile would own one serialized process; original finite producer/future custody would span queue wait, first-use boot, fixed read-only command, receipt settlement, parent postchecks and physical callback return. Parent retains its real acquisition/token and source/config/creator facts; no native scope survives a provider or human wait. Repeated cancellation waits for actual command/process settlement and cannot publish to a cancelled or stale creator.

Preserve installed application authority/source/ambient-owner checks and custom routes; reject unsupported or hidden active ownership before selection. Recheck captured installed app/config module and app-class identities, profile environment/cwd/epoch and exact selector/execution selection after serialization/startup/IPC waits, preserving final original checked provenance. Do not recursively freeze global config generation/cache or app_config values that the fixed command never consumes: original selected paths, fresh selector-file/registry observations and source-specific guards retain their checks, while ordinary concurrent settings publication remains usable (Task37 t37configb). Child Python native/audit calls execute in the child interpreter; arbitrary parent audit-hook forwarding and a global audit-inventory framework are not selected. Uncertain retirement or selected source/transport failure cannot replay as ordinary negative evidence.

Tasks36–38 keep production routing disabled through custody/fixed-command and lifecycle/selection qualification, then measure complete warm public acquisitions with idle and active parents and the unchanged original journeys. Include queue, initialization, guards, IPC and retirement; boot is outside journey time only when actual normal startup owns it. All three sequential candidate pairs must satisfy the unchanged under-five-second README/edit target and integrity assertions. This architectural selection claims no result and leaves AC5 open.

### TASK-40608 — complete key-command preparation custody (2026-11-14)

The [selected disabled key owner](../../Docs/superpowers/specs/2026-11-14-task-40608-project-index-key-owner-design.md) requires replacing the actual shared live/preview bare `to_thread` handoff in `console_provider_composition` with one `run_preparation_read` for qualified stock callers. Register the existing controller/runtime observer sets before submission and retain the complete original callback through parent status/coverage, the compatibility-conditional final guard and outer registry close. A child-command lifetime cannot stand in for that full callback. No second executor or async job ledger is introduced.

Capture the requested session and its incarnation/binding/workspace/lifecycle plus exact current controller/runtime/store and used source references on the loop; bind the lazily created index service/factory inside the retained worker. Pure-memory currentness runs at existing worker boundaries, with an explicit issuing-loop checkpoint before result consumption. Viewing another chat does not revoke an otherwise valid owning session. Do not freeze unrelated settings or add native guard calls to the loop predicate. Custom/unmanaged/missing-session routes retain their original behavior and cannot select the child.

After executor submission, caller cancellation does not retract queued or running physical work. Repeated cancellation drains that same callback before propagating, while unresolved child/native custody remains separately registered. The child owns original key-read native checks and its process-local audit execution; approvals, root guards and parent adoption stay with their original owners. Maintenance and profile shutdown close intake and prove both callback and child retirement before claiming drain. Performance remains subject to the unchanged original journeys and absolute five-second gate.

Task54's reviewed four-arm diagnostic (2026-11-15) rejects enabling the key-only child route: both first-use calls and both idle warm calls regress, while active warm caller improvements are only11.2% and4.1% in this bounded sample. All12 callbacks preserve identity, parity and physical retirement. Retain complete preparation-read custody; leave process routing disabled. ADR126 records the absolute values and limits. A larger replacement must remove meaningful parent work or repeated validation at a legitimate boundary and account for startup explicitly; first-use cost may not disappear into benchmark setup. This result supplies no original README/edit or five-second qualification.

### TASK-40608 — Console service composition and execution ownership (2026-11-15)

The [reviewed service contract](../../Docs/superpowers/specs/2026-11-15-task-40608-console-execution-service-design.md) replaces the rejected per-key process direction. First factor the existing constructors in process, retaining real App/source receivers, callback identity, lazy construction and disposal semantics. Only later introduce the explicit domain-owner/presentation-port map. The detached route has one authoritative backend ConsoleRuntime and a separately named parent ConsoleExecutionClient; the parent projection is not a second store/runtime. Legacy/custom profile construction keeps its original receivers.

An always-serviced framed reader/control dispatcher is independent of retained turn/native work. Decisions, cancellation, commit-plan acknowledgements and source fencing remain serviceable while a turn waits. Received, Applied and Retired acknowledgements have different meanings. Pressed input ordering and original decision/FIFO/cancellation rules remain; original parent attachment and backend source identities gate their respective effects. The specification defines commit-plan reservation/recovery, one stock permission publication/grant owner, source-change linearization and the finite receiver migration map. No native fence spans a transport wait.

Profile route selection is fixed before graph construction. Declining unsupported live construction must preserve all unsaved inputs and grants without initiating restart; supported live configuration/binding/revocation and unrelated screens remain unchanged. Production routing remains gated on complete caller benefit and every original README/edit candidate below five seconds in three sequential baseline/candidate pairs. In-process factoring or design approval supplies no such evidence.

### TASK-40608 Task61: retained local display and settings phases (2026-11-15)

The selected execution-service extraction uses fresh physical display phases around every original parent-only character checkpoint. One logical FULL retains its sealed character → context → appearance section set and original observers across lease-free gaps. Each metadata-pair midpoint, scope-owner/error-classification check and post-groups check keeps its original order, branch, exception and three-attempt budget. Native connection/core/producer ownership must positively retire before a parent checkpoint wait; the next phase obtains fresh admission and source checks. No section publishes before all issued physical work settles; admission or retirement uncertainty renews none. The existing same-handle legacy composition and borrowed transactions remain unchanged. This explicitly qualifies the earlier one-callback batching permission; it is not a claim that the new phase boundary was already present.

The backend owns actual domain sources and the existing authoritative display state; the actual attached view owns attachment/visit/teardown and presentation checks. No fake Screen/App, copied authority, second cache, generic callback transport, eager warm, longer reconciliation interval or skipped original midpoint is permitted. Parent archive admission retains its real App inflight/generation/cache owner, bracketing the complete original native archived-state read with fixed checkpoints after physical retirement.

Selected session settings retain one exact pending operation through the original live pre-write → full metadata snapshot attempt → settings-revision/context post-write sequence. Short preparation-lock phases surround the original retained native writer. Same-session coherent capture waits asynchronously; Cancel, decisions and source fences remain dispatchable. Exact fork-transition and global voice admission ownership remains balanced, without thread-local admission across await. The existing session lifecycle orders full snapshots with component CAS drain. Writer exception/False preserves original applied live state and does not certify durable success. The source effect includes required post-publication and exact native settlement; legacy synchronous composition remains available on its existing route.

See `.superpowers/sdd/2026-11-13-task-40608-sub-five/task-61-display-owner-plan.md` for the finite owner/phase inventory and the canonical execution-service design for the release contract. Local causal qualification is required; no process route, production selection or sub-five result follows from this amendment.
Task61 bounded convergence clarification (root ruling 2026-11-15): readiness refresh's original post-native pristine-default convergence remains before readiness key recomputation/publication. Fixed parent prepare/finish checkpoints retain the actual session controller's pristine memo, convergence spans, explicit-default generation, projection guard and provider notification. The backend performs only the exact captured automatic settings replacement through the retained pending pre/native/post owner, revalidating original session/incarnation/settings revision/config/source. Preserve follows_saved_defaults, settings=None, persona preservation and all original short-circuits (ADR095 policy unchanged). Legacy composition remains synchronous. A convergence nested inside the current ReadProjection uses that command's actual source inline; it cannot enqueue behind itself. No native lease spans a checkpoint, no second memo/span map is created, and failed/stale phases cannot publish a success/span/notice.
Task61 selected parent-settings continuation clarification (root ruling 2026-11-15): actual early FULL core-state sync and autonomous readiness refresh can reach original default convergence without a user command. They must acquire an explicitly registered ParentSettingsCapture from the same live controls/port/runtime graph, with exact real parent stamp, source revision/session/incarnation and a fixed named settings purpose. They do not invent ControlIdentity or alter external sequence/require_control_source. Only the settings resolver/effect path accepts this private registered owner; provider/tool/MCP/control authority remains unchanged. Reuse existing same-session capture tails, finite effects, task custody and settings pre/native/post operation. Register before exposure, revalidate after waits, and retire/unregister only on actual settlement, including no-entry/scheduling failure. Command-owned nested convergence remains inline. Preserve early core-state sync before later main-display/archive/readiness preparation; no early readiness warm.

Selected synchronous prompt/provider actions must defer the complete owning action when settings are pending; async actions wait before coherent capture. Capture original arguments/origin/session/incarnation/attachment, preserve action order without action coalescing, and retain original stale/refusal/notice outcomes. Pure render work retains one owed replay under its original coalescing policy. No mixed-state reads, copied settings fallback, arbitrary callback dispatcher, native lease across wait, or success before completion. Legacy synchronous composition remains original. Exact entrance map is task61-settings-parent-entrances.md; normal production selector/process integration remains outside Task61.


Task61 fix2 due-live-read and complete-core continuation (root ruling): the original early FULL notice and nonblocking config-lock/defer-once/refusal-episode gate decide whether core is due. Autonomous FULL retains that complete fixed phase in its exact ParentSettingsCapture Task, including any due original readiness read, convergence, and original core push; genuine command FULL stays in its retained command Task. Share the original synchronous gate/refusal/core bodies with legacy. No child-bound result escapes to another Task. Later display/archive/readiness order and original CORE/Canvas handoff retirement stay unchanged.

Only after an actual ensure/live-read due point, register a one-use continuation under the exact live command/parent Task and existing stock readiness producer/handoff. The original native reader still owns both config locks, participant admission and physical retirement. Carry its exact checked result/source/key and original validity interval through original convergence; consume the resulting exact session/settings revision once in the owning synchronous ensure phase. Reuse original proof/source checks, not display-cache authority; do not probe live config a second time, invent cache installs/age, or warm before core qualification. A synchronous derivation scope may share that one completed ensure within the same uninterrupted original phase; it never spans an await. Invalid TOML can remain cold and still reach original fallback. Revoke on cancellation, owner/source/definition/visit change or completion. Exact fixed phase definitions and source/session/incarnation/settings binding remain required; source failure is distinct from ordinary stale-origin/refusal. Cost if wrong: premature/stale core publication, repeated cold reads, authority lifetime escape or user-order bypass.


Proven due-read expiry reconciliation (stale-core-a1,1.437000s before ensure): the existing one-second bound remains unchanged. A named action may retain its exact Task/effect and ask the same original readiness owner for a fresh physical read after a specific ordinary-expiry outcome; it never retries owner/source/definition refusal or replays an already completed settings write. The complete core phase instead keeps its original refusal/whole-replay/3s unreadable episode on expiry. No second synchronous re-probe, retry cap, new age or forced cache install. Retire the expired request association before issuing its fresh read; cancellation retains original entered native/post custody.


Guarded prompt transaction compensation ruling (implementation pending): keep original composer checkpoint→draft→component write order in one exact parent transaction Task/effect. Only an exception escaping that original mutation transaction triggers its fixed compensation; original False, converted-to-False writer failure and display-only failure never roll back. After the first operation is physically confirmed retired, exact transaction-owned resulting state/revisions/source/session/incarnation/binding/definitions may authorize old-session backend draft/settings restoration under the same effect. Do not overwrite intervening changes. Composer/undo/UI restoration additionally requires original composer+visit; use retained original checkpoint data for old backend restoration, never replacement-view state. Retirement uncertainty or changed source/owner cannot authorize another write. Library claim acknowledgment follows full transaction/compensation; legacy sync bool stays original. No generic rollback authority or external command.


Task61 K1 prompt resolution completion: capture original flow/session/incarnation/source revision/attachment before the original provider resolution await. Provider work stays with its original owner. Its returned resolution is request data only. A fixed registered `prompt-resolution` capture joins existing same-session order, revalidates that original origin and exact flow definitions, then runs the original activation disclosure/pin, manual pin, or request-snapshot completion. Original live System/session stale check precedes any pin/publication; if that exact check reaches the original due-default boundary, use the existing one-use read/convergence continuation inside this same capture. No settings lease across provider I/O, arbitrary callback, copied settings fallback, or success from a stale resolution. Legacy completion bodies remain shared and synchronous. Provider failure may produce its original unavailable disclosure only after the same selected origin check.


Resolution-a1 confirms the matching initial coherent phase is also required: manual resolution reaches original build-provider-selection→ensure→busy-default before provider I/O. The same finite prompt-resolution purpose therefore has fixed prepare-activation/manual/snapshot phases (original stale/projection/pin checks and provider selection) and fixed completion phases. Each phase joins original session order under a real registered capture; provider await remains between phases with the exact original origin retained. A due/default read stays inside the actual phase Task. No lease, Task authority, or cached settings is carried through provider I/O; source/session/visit is revalidated before completion.


Task61 stock eager scheduler custody ruling: parent capture body waits on one private startup Future before capture_entered. The scheduling owner installs the exact actual Task in the capture/registry/custody, then releases that one-use gate. This preserves stock lazy and eager factories without changing the application factory or weakening Task checks. Actual inner cancellation while the eager body is suspended at the gate is no-entry: gate may already be cancelled, so release must not raise InvalidStateError; existing no-entry effect/admission cleanup and unfinished predecessor relay still decide retirement/order. Scheduling refusal closes the exact unstarted coroutine. Native-entered work retains unchanged physical/post settlement.


Task61 realtime completion closure: original connect keeps its API-key/model/voice reads in their original pre-instructions order, then a fixed `realtime-instructions` parent phase reads only session instructions under the exact real realtime controller/session/connect_attempt and selected settings/due owner. Original provider construction/connect remains with the transport worker after exact attempt revalidation; this phase grants no provider authority. The existing `realtime-ready` phase retains original history/instructions seed→tap release→ready/FSM publication, and handles its original due/default boundary within the same retained capture before any seed or ready success. Instructions/seed/marshal best-effort catches rethrow selected pending/due/origin/source signals; ordinary legacy errors keep their original fallbacks. Session/attempt/source/definition drift refuses, without ready-without-seed.

Realtime source edge clarification: before any selected instructions/ready settings invocation, validate the original wiring factory definition and exact ensure/runtime/store lambda code, globals, defaults and sole captured Screen. Matching returned runtime/store values alone cannot certify these callback edges. Legacy/custom routing remains unchanged.

Realtime entrance discovery uses the original wiring-installed Screen association before invoking any selected ensure/runtime/store callback. Unassociated custom controllers retain the existing legacy accessor route; a selected runtime without that actual association refuses. Association is rechecked against the actual attached controller and defining wiring factory.

Task61 attachment completion closure: capture original actual session/incarnation/source revision/visit/composer before clipboard or file I/O. I/O stays with its original worker. After each result, fixed attachment completion phases join the existing session order, validate that target and source, and run shared original clipboard-stage/file-inline/file-stage bodies; due/default ensure remains inside that actual capture before mutation. Clipboard path batches retain the original cap and attempted-child count (the child still returns None), revalidate the original target after each child await, and never continue into a replacement chat. Pure control-bar replay keeps its existing policy. For an original no-active-session entrance, execute only the unchanged creation checkpoint synchronously after exact origin validation; if it creates a target and reaches the typed due boundary, bind that exact just-created target without an intervening await and continue under the registered capture. No early session creation, copied settings, arbitrary callback or held lease across processing.

Guarded Prompt implementation checkpoint (approved fixed compensation ruling): factor the existing guard, composer/draft mutation, legacy rollback, and post-warning bodies so the synchronous compatibility methods retain literal Boolean results. The selected modal fast/dialog callback and Library consumer own a registered guarded-prompt continuation; Library acknowledgement follows its actual transaction/compensation. A fixed transaction record retains the original checkpoint/text and phase-result composer checkpoint, exact draft revision/authored token/text, and settings object/revisions. PendingSettingsApply records actual Task, whether native entry began, its original exception, and owned locked mutation state; these facts support only the same transaction's exception-only compensation, never generic rollback. The first operation must be conclusively no-entry or physically confirmed retired; source/currentness failure, uncertain retirement, or intervening draft/settings/composer state refuses compensation. Backend old-session compensation uses original setters and fresh counters after view switch; UI restore/publication separately requires the original visit/composer and unchanged transaction-result checkpoint. A known physically settled failure is recorded as failure, not fabricated commit; any fixed pending-operation handoff/retirement must balance the exact original owner. Writer False/converted False and display-only failure never compensate. Original prompt callback/store/composer defining edges are checked before selected invocation; unrelated legacy/custom callbacks keep their original route.

Guarded Library claim lifetime: the fixed retained owning action includes its original post-transaction acknowledgement. Observer cancellation cannot release an admitted completed/compensated mutation for retry; ACK runs in the same actual retained Task before observer cancellation is re-raised. Proven pre-transaction/no-entry failure retains the original exact-claim release-for-retry behavior. Uncertain physical settlement cannot fabricate ACK, rollback, or a retryable completion. The new guarded source file is an explicit required inventory entry with its saved absent preimage.


Task61 fix3 authored-draft clarification (2026-11-16, root ruling): the selected synchronous authored publication keeps the original set_session_draft and external handoff callback order. Its only later read is draft_revision: after setter return, read it under the existing preparation lock from the exact returned still-registered session, preserving missing-session False and refusing replacement identity. Legacy full input snapshot remains unchanged. No new outer lock spans the original external handoff callback. For the guarded transaction only, retain expected authored text/token/revision before each original setter and verify the same returned session after callback; callback/intervening changes cannot be sampled as transaction-owned rollback state. The fixed result witness is data within the already registered exact Task/phase, not settings/native authority or a generic callback. The original composer phase and explicit draft setter keep their order; compensation requires the exact resulting composer checkpoint and draft witness. Same-view edits during a native hold must succeed as original draft edits and then prevent compensation from overwriting them. Costs if wrong: mixed-state exception during typing, callback deadlock, or lost newer draft/undo history.

Task61 fix3 prompt target continuation: the two asynchronous /prompt insertion owners retain original target capture before search/picker work through a fixed prompt-target parent capture. Its original _capture_prompt_replace_target guards/ensure run in same-session order; a due boundary uses the existing exact checked-read continuation and revalidates before target publication. No arbitrary function payload or settings copy. Legacy synchronous target behavior and original no-active-session creation point remain. This closes pending-only entrance waits that could otherwise reach cold/default ensure unowned.

Task61 fix3 guarded actual-write cancellation clarification (root ruling): after draft admission, only the exact first durability Task positively proven canceled before apply/body/native entry may enter the fixed compensating phase after that Task is joined. Original CancelledError is retained after original compensation/False warning and terminal Library ACK; the admitted claim is never released for replay. An entered operation with confirmed native/post commit instead completes the transaction/ACK and preserves cancellation without rollback. Uncertain entry/retirement and changed authority/result state cannot authorize compensation. This repairs the introduced asynchronous transaction boundary, not a general BaseException rollback rule.

Task61 fix3 no-active prompt target: preserve the original ensure creation point and ordered-resume no-creation result. If that original ensure creates/returns its actual session and then reaches typed selected due, attach the original controller/store/session/incarnation/settings/revision/current-Task witness at that exact ensure frame. The target adapter captured source/revision/visit before the call and validates that witness and the same active registered session after it; an intervening replacement refuses. It may then admit only the same fixed prompt-target continuation. The exception data never grants native authority; current real graph/port/source and normal finite effect admission still do. No adoption of whichever session is active in the catch, no early creation/warm or new persistence.


Task61 fix3 guarded post-display convergence: guarded-post-a2 proves original completed System transaction -> original cold due request -> actual default convergence -> stale transaction-witness refusal. The fixed completed-post phase may advance only its settings-result witness through the exact original convergence operations admitted during that same due-read continuation. Each operation records its locked pre-live state, original native-required decision, and committed/post resulting state. Provenance is issued at the actual producer return while its exact waiter/source/handoff/key and any original B dependency association are retained. Expiry-only fresh reads retain only that same call's operation chain; no age extension, source retry, repeated completed write, or authority from settings equality. Validate the final registered one-use result plus every predecessor/result link, successful actual Task/post and positively confirmed native retirement (or positively proven original no-native decision), unchanged source/session/incarnation/binding/visit/draft/authored token/composer. A readiness B remains its original source/effect, never relabeled A. This is post-display completion only: no rollback-baseline revival, compensation authority, false Applied, or suppressed selected refusal. Cost if wrong: accepting intervening state or duplicating a completed mutation.

### Task62 fixed process receipt reconciliation (2026-11-16)

The approved whole-service transport uses a pure closed codec and independent
reader/serialized writer, retaining original control and physical owners in the
backend. ApplySettings transports the original submission and detached accepted
LiveCommit (including its normalized accepted submission), never the local
durability lease or an invented durable-success bit. ReceiveTurn additionally
carries exact press values; revision-only sampling of later backend input is not
an equivalent remote command. Original stage-specific input and attachment limits
remain where their original owners enforce them; the framing chunk is not a total
payload ceiling.

Source/publication/review values refer to the exact producing request in a
backend-owned registry. They are correlation, not transferable source/permission
capabilities. A pre-claim refusal terminates the Received observer with the real
failure rather than fabricating claim adoption. After a claim is applied, later
turn failure stays a separate terminal outcome. Sending a matching retained plan
ACK makes the parent potentially committed before delivery can be known. Channel
loss, process death and containment cleanup do not certify native retirement.

The original local presentation port remains exact. A separately qualified
backend endpoint carries only named parent checkpoints and detached projections;
it is not a fake App. Parent attachment/visit checks remain actual parent facts,
backend source/native checks remain actual backend checks, and no physical lease
spans a checkpoint transport wait. Task62 implementation and whole-path evidence
remain required before any production selector or performance acceptance.

### Refusal of the display check before a commit acknowledgement

The parent must run the original presentation check before acknowledging a
planned commit. The backend waits for that acknowledgement while it owns the
reservation. When the current visit receives a verified, positively retired
refusal of that display read, cancel the exact still-pending turn through
the existing `CancelTurn` protocol. Join the cancellation and original turn
receipts, including the terminal custody callback and positive retirement,
before reporting that the draft was kept. The original check still refuses.
The writer must never start on this path.

This settlement requires the original client, source revision, visit,
session, incarnation, retained plan and an unsent acknowledgement. An
acknowledgement that may have reached the writer, channel loss, or missing
retirement evidence leaves the outcome uncertain. The client cannot infer
success from any of these conditions. No plan or action is replayed.
