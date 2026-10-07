# ADR-222: Console Send preparation and I/O ownership

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
