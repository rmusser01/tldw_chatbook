# ADR-084: Make Change Review explicit, asynchronous, and advisory

Status: Accepted
Date: 2026-08-21
Related Tasks:

- [TASK-19501 - Make Change Review opt-in per workspace](../tasks/task-19501%20-%20Make-Change-Review-opt-in-per-workspace.md)
- [TASK-19502 - Decouple Change Review finalization from Console turn completion](../tasks/task-19502%20-%20Decouple-Change-Review-finalization-from-Console-turn-completion.md)
- [TASK-19503 - Bound Change Review baseline gating before tool dispatch](../tasks/task-19503%20-%20Bound-Change-Review-baseline-gating-before-tool-dispatch.md)

Supersedes: the default-on consent posture and synchronous turn-finalization
portions of
[the 2026-08-02 Agent Change Review design](../../Docs/superpowers/specs/2026-08-02-agent-change-review-design.md).

## Decision

### Workspace consent and retained content

Change Review is opt-in per workspace. The global `[change_review] enabled`
setting remains a master capability switch: a missing global setting keeps the
capability available, an explicit false disables it, and an unreadable or
uncoercible value makes it unavailable and fails runtime tracking off.

Workspace consent is stricter. Only an explicit stored true enables tracking;
a stored false, missing row, or storage failure disables it. The registry
returns a typed enabled/disabled/unavailable result plus an opaque durable
revision for available reads. Toggle writes compare both the observed state and
revision in one transaction. Each successful write receives a distinct
revision even if the clock is frozen, so a disable/re-enable ABA cannot satisfy
an old compare-and-set.

One app-owned consent service uses memory-only workspace revision fences around
local admission, toggle commit and initializer readiness publication. Registry,
capability and durable CAS calls run outside its bookkeeping lock. A turn
admitted before a disable commit may finish; every admission after that commit
is untracked. An initializer may publish `ready` only if the workspace still
has the exact enabled revision it captured. Enabled roots initialize in the
background and expose preparing/ready/failed state with bounded retry; chat
never waits for initialization. Preparing and failed roots are omitted from
that turn with a visible untracked warning.

Settings states that Change Review stores shadow Git history under application
data, including file contents, for the configured retention period (30 days by
default). Disabling prevents future admission but does not erase already
admitted or retained history. History deletion is a separate decision because
one canonical root can be shared by workspaces and active review operations.

### Ordered asynchronous finalization

Review work does not own Console chat completion. After the assistant outcome
and terminal persistence settle, the bridge schedules review finalization and
returns. The prompt coordinator releases the accepted turn and may drain the
next prompt without waiting for filesystem or Git work.

One app-owned `ChangeReviewFinalizationCoordinator` owns review windows. It
uses a fixed worker set, a bounded pending-operation queue, and FIFO lanes per
canonical root across conversations. Registration reserves every root or none;
partial reservations become tombstones that lane workers skip. A lane covers
the full baseline-to-end window and preserves the existing survivor-child
boundary before a successor window begins. Filesystem workers receive immutable
snapshot inputs and original host lifetime context, and return values to the
publisher. Native workers enter fresh original storage/source scopes on their
own threads; no SQLite connection or entered scope crosses the handoff. Widgets
and UI callbacks remain outside filesystem workers.

Review publication is a durable join, not an in-memory handshake.
`AgentRunsDB.agent_runs.assistant_message_id` supplies the assistant anchor and
`change_snapshots` supplies review results. Either writer may commit first and
request the same idempotent refresh. Rendering inserts a marker immediately
after the assistant only when both durable facts exist. An unmounted screen
derives the same result at the next mount.

Shutdown stops admission and scheduling, bounded-drains already-running pure
workers, invalidates late generations, disposes the coordinator, and only then
closes `AgentRunsDB`. A worker that outlives the bound may finish shadow-repo
filesystem work, but its result cannot persist or touch UI state.

### Native Git physical completion (Task 8D2 Git custody)

Selected Git may be a legitimate delegating wrapper; successful version/config
output and direct-child exit do not prove descendant completion. All three native
launches (version, config, command) use the existing low-level process-tree
controller on the original synchronous worker or publisher thread through a local
asyncio Runner. The same initially opaque ticket/source/operation remains active;
no executor, replacement owner, queue or reaper is introduced. Guarded storage and
SQLite work stays on its original synchronous activation, outside the transport
coroutine. Existing argv, environment, inherited stdin, byte/text behavior,
timeouts and helper suppression/refusal remain unchanged.

Completion joins actual stream collection, containment-empty proof and handle
closure, original worker return, original host acceptance/discard, and the return
of operation.finish before root completion and claim retirement. Borrowed tickets
never settle the covering parent's claim. Its future producer must keep that
parent root/source/storage live until the original ticket.retired; premature
parent finish is outside this producer's contract. Timeout or exceptional command return
requests bounded TERM/KILL; a still-observable tree then uses the same controller's
late wait under the original worker, source/storage scopes and stopping operation.
Delay alone is not irreversible cleanup uncertainty. If proof itself becomes
unavailable, cleanup is marked unproven and the original worker/scopes/operation
remain retained. This can hold disposal indefinitely; leader exit, readiness,
publication or a cancelled shutdown waiter cannot manufacture proof.

Only initial native Change Review attachment uses the existing
native_workflow_qualified platform gate. Unsupported platforms retain ordinary
consent/coordinator/Git behavior, while qualified source loss cannot downgrade.
Managed groups are the proof boundary, not resistance to OS-level escape; no new
Windows native qualification is claimed. Costs are process-group/Runner overhead
and retained finite capacity/storage while cleanup remains unresolved. Historical
Undo checkout/filter semantics are unchanged and remain separate work.

### Conservative, bounded pre-dispatch observation gate

Task 8D1 qualifies the app-owned native coordinated path: finite opaque native
ownership covers inherited Git config/include/ignore reads and actual snapshot
callbacks through original publication return. A shutdown deadline cannot retire
an active callback or close its original publisher storage. Never-started jobs
may cancel conclusively; admitted final publication retains its original owner.
Both failed publication attempts expose an unresolved disposition, separate from
physical retirement and any workflow acknowledgement (ADR-285).

Native capture supports Git >=2.36 with per-invocation hooks, signing, fsmonitor,
GC/maintenance and external diff/textconv suppression. Configured external
clean/process/smudge filters and promisor/partial-clone remotes make capture visibly
unavailable before helper-capable entry, including implicit object fetch.
Configuration discovery stays private; global ignore/privacy rules, existing
history/restore and user Git configuration remain unchanged. Known native opaque
coordination does not isolate external OS configuration edits. The standalone
legacy tracker keeps its existing behavior. Consent's separate root initializer
uses the original feature source and native qualification described next.

Task 8D2A supplies that initializer dependency through the existing two-worker
queue. Its separate original feature source captures actual configuration,
registry, binding, consent revision/generation, callback and shadow storage.
Each dispatched initializer takes finite opaque activity and one native tool
operation, retaining both through actual initializer and readiness/discard return.
It creates no Chat or AgentRun. Source reads, callbacks and native admission stay
outside the consent bookkeeping lock. A failed or stale result never publishes
READY; an unsuccessful observational handoff is distinct from workflow success.

Explicit Settings enable/retry uses manual capacity; lazy configuration and
binding observers use automatic capacity. Busy work uses existing failed/busy
readiness and Settings retry, including contention between opaque initializers.
Closing seals/discards unstarted work before runtime capacity closes and retains
original worker/storage custody through real completion; bounded joins remain
advisory. Chat still never waits for root preparation. The next ordinary-tree
producer must borrow its original covering opaque claim and reserve the finite
initializer hold before enqueue and parent finish, without reacquiring activity
behind itself. This stage does not implement ordinary or workflow phase admission.

D1 snapshot issuer construction does not initialize capacity settings; first actual
admission resolves the same runtime ledger on the retained publisher callback.
Late no-effect admission refusal is UNADMITTED, including discover/B. It delivers
a once-only transient marker to the original store/session/assistant without a
trajectory or snapshot publication fallback. This ordinary advisory display is
not durable workflow-window failure disclosure; that remains a D2 obligation.

Change Review remains observation, not authorization. Existing project-context
preparation, permission review, invocation-time permission resolution,
refusals, stamps, audit, and kill-switch ownership remain unchanged.

Task 8D2B1 gives real legacy hook events their own finite opaque native custody
through target cleanup and original host delivery/discard. Arbitrary hooks are
not observational snapshots and gain no snapshot authority. Required existing
durable writes remain required; an empty ordinary-event payload obligation does
not require a new hook journal. V2/MCP/SessionEnd and subsequent ordinary-tree
borrowing (including enqueue-time initializer holds) remain mandatory pending
integration, with workflow capture still refused.

For a review-enabled turn, the order is:

1. prepare tool calls;
2. run the existing combined review hooks;
3. skip denied or otherwise non-proceeding calls without waiting;
4. bounded-wait up to three seconds for the baseline across every tracked root
   for all calls that may still dispatch; and
5. enter the existing invocation path.

If the review hook raises, the conservative wrapper performs the bounded
all-roots wait before the runtime's existing hook-failure policy continues.
The only baseline-wait bypass is a fixed table of app-runtime handlers that
cannot mutate workspace files: `find_tools`, `load_tools`, `skill_file`,
`search_run_log`, `run_log_stats`, `run_log_slice`, `wait_agents`, and
`check_agents`. Provider, skill, script, spawn, install, message, and unknown
calls wait conservatively.

A timeout never blocks tool dispatch after the bound. Unresolved roots become
irrevocably untracked for that turn, late baselines are ignored, and a visible
content-free warning explains the missing attribution. If the timeout occurs
while a predecessor survivor window remains open, both lineages become
attribution-invalid and the root enters a degraded epoch. It resynchronizes
only after known survivor and mutation-capable work becomes quiescent; no late
result can retroactively claim the intervening mutation.

The visible bounded prompt queue remains process-memory-only under
[ADR-046](046-visible-bounded-console-prompt-queue.md). This decision fixes the
turn-release boundary; it does not persist or replay unsent private prompt
text.

## Context

Change Review currently starts a baseline thread for an ordinary Console turn
and synchronously joins it during terminal finalization. On a production-shaped
9,301-file root, warm baseline and end snapshots cost about 309 ms and 230 ms;
the initial snapshot took 6.62 seconds in the refreshed probe. The assistant can
already be durable and visible while the bridge still owns
`accepted_live_turn`. A third prompt submitted in this interval is only queued
in process memory, which reproduces the reported apparent "file checker" stall.

The same feature is currently enabled when workspace state is missing or
unreadable and retains shadow-repo file contents without explicit workspace
consent. Moving only the terminal call to a detached thread would release the
turn but would not preserve shared-root ordering, survivor attribution,
bounded ownership, shutdown safety, or durable publication.

These are long-lived privacy, state-ownership, concurrency, cross-module, and
application-lifecycle decisions, so an ADR is required.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep Change Review default-on | Shadow history retains user file contents; capability availability is not workspace consent. |
| Disable Change Review globally by default | Removes a useful feature rather than giving each workspace an explicit, inspectable choice. |
| Detach one end-snapshot thread per turn | Loses bounded ownership, shared-root FIFO ordering, survivor lineage, and safe shutdown. |
| Block chat until root initialization finishes | Recreates the incident on large workspaces and makes optional observation own the primary chat path. |
| Persist a marker or coordinator handshake in the Console store | Duplicates existing durable run/assistant ownership and fails across unmount/remount. |
| Let baseline failure deny tools | Turns an advisory review feature into a second authorization system and can wedge legitimate work on Git/filesystem failure. |
| Predict every tool's mutation targets/effects | Duplicates provider grammars and permission ownership; a conservative bounded all-roots gate is smaller and safer. |
| Persist queued prompt text until review completes | Violates ADR-046's privacy and restart semantics and addresses the symptom rather than the held turn. |

## Consequences

### Benefits

- Ordinary workspaces pay no shadow-snapshot or retention cost until the user
  opts in.
- A finished response releases the session promptly even when Git or filesystem
  work is slow.
- Shared-root and surviving-child attribution retains a single ordered owner.
- Result publication survives writer races and screen navigation through
  existing durable state.
- Baseline failure remains visible and bounded without weakening the existing
  permission system.

### Accepted trade-offs

- An enabled workspace may show preparing or failed tracking while chat and
  tools continue untracked.
- A configured deny enforced only at invocation may pay the bounded baseline
  wait because Change Review does not duplicate permission resolution.
- A disabled workspace writing a canonical root also tracked by another
  workspace remains an external-writer attribution limit.
- Cross-process shadow-repo integrity remains protected, but this coordinator
  does not claim a distributed total order across application processes.
- Disabling does not erase retained history; a separate deletion design is
  required.

## Verification Consequences

The implementation requires real-database state/revision/CAS tests,
barrier-controlled consent and initializer ABA tests, real-Git coordinator
ordering tests, result-first and anchor-first durable publication tests,
survivor-timeout degraded-lineage tests, shutdown tests with a late pure worker,
and a mounted three-turn Console test that holds finalization while the third
turn starts. Live verification uses an isolated profile and completes three
turns with review disabled and enabled while recording only content-free timing
and lifecycle facts.

## Links

- [Approved recovery design](../../Docs/superpowers/specs/2026-08-21-console-file-review-performance-recovery-design.md)
- [Original Agent Change Review design](../../Docs/superpowers/specs/2026-08-02-agent-change-review-design.md)
- [ADR-300: Local agent tool permission boundary](300-local-agent-tool-permission-boundary.md)
- [ADR-046: Visible bounded Console prompt queue](046-visible-bounded-console-prompt-queue.md)
- [ADR-317: Console project-instruction local state and preflight](317-console-project-instruction-local-state-and-preflight.md)

Task8D2C reuses the native Git qualification and physical transport described in
ADR285 for explicit references through an explicit executable/repository prefix
and original producer. ShadowRepo remains the original Change Review consumer;
its helper policy, storage/source/host contract, timeouts and ordinary unsupported
platform behavior are unchanged. Reference reads create no Change Review repo,
consent state, snapshot or durable publication obligation.


### Confirmed historical Undo custody (Task 8D2)

The original app Runtime binds a separate Undo issuer at bridge/provider composition. Actual Review or Console-card confirmation consumes the exact captured rows, paths and fingerprints; saved IDs and direct provider calls grant no authority. Finite OPAQUE preflight reads retire before the human modal. Confirmed single-file and multi-root Undo acquire one complete OPAQUE claim and one manual nonchild tool operation before original effect entry. Native activity/recovery/capacity conflicts refuse without a queue; active-session refusal remains additional. Original DB/conversation/row/root/source and disk fingerprints are revalidated under custody. Qualified source loss cannot downgrade to unowned work.

Historical checkout/filter and snapshot semantics use the existing native Git process-tree transport independently of D1 qualification. Once mutation is attempted, even partial per-path failure requires the original post-revert snapshot. Successful paths require exact original reverted-path-union DB readback; a committed-then-raised update can be proved this way. A snapshot exception has no supported replay or arbitrary-HEAD proof. Unknown required outcome delivers honest partial results then retains the same worker/source/storage/OPAQUE protection indefinitely; no new journal, retry producer or poller is introduced.

The original host accepts results or conclusively discards stale presentation before operation.finish RETURN; original worker activation, DB handles, shadow storage and any actual scratch-generation lease survive that return. Existing scratch lookup is metadata-only and cannot allocate: tombstoned/lost previously recognized generations refuse instead of becoming unmanaged. Admission/start fences seal unstarted work during disposal; already-started original obligations may finish on their original retained source, without new confirmation, replacement-source permission or liveness inferred from a cancelled waiter. Costs are coarse contention, exact source/readback checks and potentially indefinite disposal retention after unprovable cleanup or settlement. This qualifies only the existing confirmed Undo feature; ordinary tree, wave and public workflows remain pending.

A started thread whose original activation cannot enter may retire as unused only after all partial scopes close normally and the same thread observes operation.finish return. The effect callback has not entered; no required mutation exists. The same distinction covers sealed unstarted tickets. Callback-body failures, uncertain scope unwind and unknown finish return retain custody. No replacement-source retry is permitted. The existing preparation-read hop performs only source/DB/capacity admission before original effect-worker enqueue; its extra pool handoff is an explicit cost.


### Ordinary covering-tree integration (Task8D2)

Original consent/status and eligible-binding metadata precede actual initialization
admission; pending enabled initialization contributes to the complete maximum.
Each ordinary initializer reserves original borrowed custody before its existing
queue accepts work. Discover/B/E and survivor jobs use the same original covering
owner/claim; genuine future E/survivor holds precede parent/capacity closure.
Original callback, publication and operation.finish return precede ticket.retired
and parent release. First-unsaved-turn borrowers consume the original launch key
with the original store's committed durable identity mapping; saved rows keep real
conversation IDs. Required bundles cannot wait behind their own held slots;
original advisory omissions remain visible. No new consent, queue or journal.


### Concurrent phase-wave capture (Task8D2; approved before behavior)

ADR-285's concurrent phase-wave contract qualifies the existing coordinator.
One real B/E covers each concurrently admitted eligible task set. Original
WorktreeActivity membership orders claims and windows under the same mutex;
finite snapshot jobs retain OPAQUE physical conflict without cross-root fence
cycles. Earlier ordinary and ready manual work takes priority. Human consent and
finite preparation precede the wave; committed child claims survive tool approval.

The original coordinator issues an inert reservation before the workflow FULL
active-window commit; only the same original reservation/source/receipt activates
normal jobs afterward. B callback, handoff and physical return precede child
writes. Bind the first actual committed member run, never a fake supervisor.
Close E at actual quiescence before manual/capacity/human gaps; fresh B is required
after intervening effects. No-commit races close an honest unused window.

Workflow-only v28 window/ref records cover zero-row E, unused/no-run and late E
without changing ordinary snapshot rows or immutable task checkpoint bytes.
Publication atomically writes native rows and refs/disposition. A successful
fallback discloses failure; both failures retain required window/phase/original
source/storage/config/recovery protection and cannot acknowledge quality success.
Conclusive finite observational return is distinct from required durable success;
never label failed rows durable to release them. Unknown physical/source proof
remains retained. E handoff failure preserves a previously conclusive B.

Active/unresolved windows block pruning/removal despite review closure; settled
refs persist until permitted payload removal. Original source counting extends
through actual E/publisher/finish return. Native consent, exclusions, historical
contents, helper qualification and confirmed Undo stay unchanged. Exact bounds,
budget accounting, migration/catalog/archive/recovery and costs are recorded in
[ADR-285](285-native-phased-agent-workflows.md#concurrent-phase-wave-capture-task8d2-approved-before-behavior).


Task8 D2 close refinement (source-confirmed): workflow preparation must not allocate the unused primary model lifeline. Only `run_primary` starts it, and that entry refuses workflow dispatch; actual children retain their unchanged native child model lifelines. Launched-task native frame teardown returns under a pre-reserved operation on that same original child before its physical drain. A no-launch frame may retire only the source-verified empty display/maps and native provider unbind metadata; any unexpected effectful callback or changed cleanup source refuses, retaining its source lifetime. There is no fresh cleanup owner or OPAQUE claim against the wave, and no generic cleanup exemption.


Task8 D2 required-reference count refinement: the v28 window records a bounded snapshot_count alongside the ONE normalized ref table: NULL while required E is pending/unresolved, 0..64 atomically with conclusive E rows/refs, and zero for unused/removed windows. This is the necessary count witness distinguishing a genuine successful zero-row E from missing required references during restore validation; it carries no second mapping or authority. It remains inside the existing 65536-byte reservation and does not change v25–v27 catalogs.


Task8 D2 original native failure/host return refinement: a known failed native create/setup/thread start must finalize its original log and return its original frame under the pre-held close operation when no child thread was started. Unknown creation retains custody. Reserve one additional finite launch-host operation on the SAME child owner before dispatch callbacks, and finish it only after the original launch source/storage/execution scopes return; it prevents an early child or known failure root finish from releasing the original claim before host return. It adds one existing operation per admitted child, not an owner, scheduler or claim. Tool-worker automatic budget/authority checks must enter the existing owned_goal_storage scope before their first database read; the actual child model-loop coroutine must activate its exact original workflow source and owned_goal_storage before gateway checks. The unchanged original lifeline and pre-held child close/join retain protection through these real thread/source/storage returns. Live native archive evidence found AgentRunsDB leases on departed tool/model threads when those checks preceded the finite scopes; no cache reset or cross-thread close substitutes for their original return.


### Wave review fix1: receiving source, unused publication and native callers

Task8 D2 wave review fix1 (FIX_BASE c441672e4cffeb0310df9c133bd527ef883438f8): retain the actual native local provider base specs, admitted roots, alias-specific receiving specs and native handler function records after composition using the existing ordinary capture/check helpers. Revalidate through the existing post-permission/final task provider checks; replacement refuses before effects, without a fabricated ordinary tree, pruning, permission change or post-admission widening.

An original active-window body refusal is conclusive only when the same recorded body exception escapes the existing FULL transaction after successful rollback, PRAGMA restoration and original connection/storage return. A window-local private outcome binds that proof to the exact store, phase and inert reservation. Consume it only after the original launch source/storage scopes return. The same coordinator lock verifies original registering/no-operation state and removes only that inert reservation; original unused-source retirement executes outside the lock and must return, then each native no-launch frame must close before fence/configuration release. Commit/ack/unwind/source/cleanup uncertainty retains original protection; no inferred absent row, generic exception release, phantom unused row, schema change or ordinary required-write weakening.

The existing native-only task route remains supported without a Console frame. Its original bound owner, allowance, path/tool checks, native child preparation, fleet, worker/model scopes and cleanup remain authoritative. Only an actual original Console preparation adds wave anchoring, original-source wrappers and pre-held frame-close/launch-host operations; never create those operations for an absent Console caller. Audit bind, tool, launch, model/finalizer and no-thread failure together. ADR required: existing ADR285 and ADR084 apply; no new ADR. Cost: retained concrete instance records, one private no-commit outcome, exact inert cleanup and optional-frame checks. Native negative controls and existing native-only launch/cleanup tests qualify the changes; Task38531 remains In Progress, with Task8E/F/G/WHOLE outstanding.
