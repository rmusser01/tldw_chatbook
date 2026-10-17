# ADR-230: Console Send preparation and I/O ownership

Date: 2026-10-06
Status: Accepted following written review and the requested issue audit.
Task: [TASK-34563](../tasks/task-34563%20-%20Design-Console-Send-preparation-and-I-O-ownership.md)
Spec: [Console Send preparation architecture](../../Docs/superpowers/specs/2026-10-06-console-send-preparation-architecture-design.md)
Extends: ADR-094, ADR-098, ADR-126, ADR-148, ADR-163, ADR-197 and ADR-220.

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

ADR-094's app-owned accepted-turn contract is extended earlier to received-intent custody. Complete request construction and durable acceptance remain separate. Navigation still detaches presentation; initial review can now remain app-owned.

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
