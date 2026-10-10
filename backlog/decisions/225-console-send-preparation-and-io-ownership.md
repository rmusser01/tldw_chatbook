# ADR-225: Console Send preparation and I/O ownership

Date: 2026-10-06
Status: Accepted following written review and the requested issue audit.
Task: [TASK-34563](../tasks/task-34563%20-%20Design-Console-Send-preparation-and-I-O-ownership.md)
Spec: [Console Send preparation architecture](../../Docs/superpowers/specs/2026-10-06-console-send-preparation-architecture-design.md)
Extends: ADR-094, ADR-098, ADR-126, ADR-148, ADR-163, ADR-197 and ADR-220.

Renumbering provenance: this unmerged Console proposal used number 222; PR #3050's rebase preserves dev's landed provider HTTP session reuse decision at that number.

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

## Hook authority read sharing clarification

Decision 3 applies to hook consent as follows. A Send attempt's first full consent read (received-intent preparation, otherwise the first submission admission) is passed explicitly to that attempt's own submission admission, v2 hook preparation and legacy UserPromptSubmit target selection, which use it only while it matches their session and owner and nothing the process has observed has moved it (store revision, section stamp, profile, refresh, seals, close, config identity); otherwise they read fresh. Even a standing shared read answers only "nothing to do": an admission it does not refuse, no UserPromptSubmit hook to select, no v2 hook to prepare (no handler configured or granted, no plugin-owned skill, no retained session hook state). Any hook to select, build or keep is chosen from a fresh read: a change made only by another process after the shared read is invisible in memory, and a target selected from it would reach its fresh launch guard stale and be refused, blocking the Send, where a fresh selection simply omits a disabled or removed hook. The read is never stored by the owner, time-based or shared with another attempt. The final pre-dispatch admission stays a fresh read, and every hook launch keeps its fresh launch guard. Consequences, both limited to a change made only by another process (or a hand edit) between the shared read and the consumer: a change a fresh pre-commit read would have refused (a newly enabled unreviewed hook, an unavailable store) is refused at the final admission, after durable commit and retained for recovery, rather than before commit; and a hook enabled and approved elsewhere in that window does not run for that attempt.

## Builtin-only composition refinement (OPT-88, 2026-10-09)

The user explicitly approved making a builtin-only Send independent of unused external-catalog errors and definition-change audits. After the initial fresh tool maximum is captured, stock live composition skips external catalog reads, admission, migration and external definition/audit work when its exact frozen tool-ID ceiling contains only the builtin namespace (or no tools). This is a deliberate behavior change: corruption or changed definitions confined to the unused external catalog no longer delay or refuse that composition and are observed when that family is next needed.

The initial maximum capture remains fresh. A missing, unknown, mixed or external-capable ceiling retains fresh external observation and its existing errors/audits. Builtin inventory and participating consumers' common permissions/kill switch remain fresh; policy uncertainty still refuses the affected operation. Actual invocation retains all live gates. Both shared preparation and the stock ordinary fallback honor the same dependency decision; custom adapters retain their documented route. No cache, delayed audit queue, new persistence format or wider tool authority is introduced.

Alternative considered: always read and audit excluded external definitions to preserve incidental side effects. Rejected for builtin-only composition because those definitions cannot enter the attempt and the measured empty catalog alone costs .264/.378 seconds in warm preparation. This does not promise an equivalent whole-Send saving. TASK-34601 and the OPT-88 plan require targeted controls and sequential native comparisons.


## Composition-owned stock tool ceiling (OPT98, 2026-10-09)

Under the user's authorized architecture review, qualified stock owned capture
records explicit MCP demand rather than reading policy/catalog to establish an
early ID/hash ceiling. The existing execution consumer's fresh composition issues
that ceiling once, including a successful empty or killed result. Its provider
cannot widen on subsequent composition. Invocation/revocation/kill-switch gates
remain live; metadata never authorizes execution. Explicit captured maxima and
custom/synchronous entry routes preserve their existing narrowing contract.

This supersedes the initial-fresh-maximum requirement above for that stock route.
Newly enabled/discovered/redefined tools between Send and composition can enter
the new catalog; required source errors and definition-change audits are observed
at composition, which may follow durable acceptance. Existing postcommit refusal/
recovery policy applies. Failed saving still retains the draft and refuses dispatch.

The boundary is per existing execution consumer, not a shared receipt-time ceiling
across the prospective hook executor, live agent run and disposable previews.
Each independently composes from current state; a preview grants no live authority.
Constructing a new consumer repeats preparation. Within a dispatched provider,
IDs/hashes remain frozen. Plugin snapshots retain their independent narrowing;
actual hook/tool effects keep every existing authority check. Unknown/custom entry
routes use their original capture; drift to an unsupported adapter after deferral
refuses rather than silently broadening. Temporary sessions remain MCP-excluded.

Alternative: a reusable versioned definition snapshot on LocalMCPStore. Deferred:
hand edits, other processes, recovery and cold empty snapshots lack a complete
invalidation contract. It adds ownership without removing the duplicated stage.
Alternative: retain one resolved pair across every attempt consumer. Not selected:
these already have distinct execution/preview lifetimes and fresh gates; no new
mutable attempt registry is warranted. No native custody crosses awaits, no TTL,
permission cache, new settings or persistence format is added.

Implementation/qualification: TASK34601 AC21 and
[composition tool ceiling plan](../../Docs/superpowers/plans/2026-10-09-console-composition-tool-ceiling.md).
Adoption remains subject to integrated controls and sequential native timing.


### OPT98 qualification

The composition-owned route is retained after259 distinct targeted cases and
source-stable sequential full-profile ABBA. Warm mean3.031 to2.718s and cold4.885
to3.945s are small-sample application timings; all12 turns save and settle.
Invocation, custom/captured/plugin narrowing and save-failure draft retention
remain covered. The exact results and excluded overlapping initial baseline are
in the linked plan. The broader latency/physical-feedback targets remain open.

### Merge-time captured press integration

The integrated Console preserves dev's captured-at-press Send behavior. The
existing pending press retains its original resident session-input snapshot and
typed draft through acknowledgement paint and deferred replay. Stock received
custody uses that same body and revision. Subsequent composer edits alone do not
cancel that explicit press; session incarnation/binding, settings, configuration,
attachments, prefill, staged evidence and sealed Stop remain currentness gates.
Direct intake without the explicit pressed snapshot keeps its strict draft check.
Classified text which differs from the raw pressed body keeps its ordinary route;
stale frozen source inputs refuse and cannot become a legacy fallback.

The acknowledgement remains view-only and binds its admitted/terminal handling
to the one received runtime custody. Saved or queued acceptance owns draft
consumption. Its original revision protects newer text, while the existing typed
consumer spends the capture and removes proven same-generation appended prefixes.
Hidden replaced/retyped or unknown-generation text is retained. Save failure,
Stop, source change and commit failure retain the existing refusal/recovery rules.
The new pending USER row is a separate visibility endpoint from the stored echo;
pre-rebase stored-echo timing does not qualify first visible submitted text.
