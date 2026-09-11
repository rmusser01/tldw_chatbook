# Queued Console prompts delegated to named agents

- Date: 2026-09-08
- Status: Proposed; written design ready for user review. No application implementation is included.
- Design task: [TASK-32472](../../../backlog/tasks/task-32050%20-%20Specify-direct-delegation-of-queued-Console-prompts.md)
- Decision: [ADR-137](../../../backlog/decisions/137-queued-console-agent-delegation.md)
- Existing contracts: [ADR-046](../../../backlog/decisions/046-visible-bounded-console-prompt-queue.md), [ADR-069](../../../backlog/decisions/069-console-project-instruction-local-state-and-preflight.md), [ADR-134](../../../backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md), [ADR-135](../../../backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md).

## 1. Outcome and confirmed choices

Users can assign an individual queued Console prompt to Main agent or a named
agent. A delegated entry launches through the native fleet without a main-agent
model call. Once accepted, its child works in the background and the queue can
advance. The child receives the task text, its agent instructions, and the
ordinary permitted execution context; it does not receive conversation history.
Its completion becomes eligible for the existing bounded supervisor wake flow.

The user confirmed these choices during brainstorming:

| Question | Confirmed choice |
| --- | --- |
| Who chooses the executor? | User chooses a target per entry. |
| How does delegation execute? | Direct queue-to-fleet dispatch. |
| Does the queue wait for the child result? | No; launch and continue. |
| Who handles the result? | The main agent through a completion wake. |
| What conversational context does the child get? | Task text only. |
| What happens when child capacity is full? | Keep the entry at the front and resume automatically when capacity becomes available. |

The user subsequently requested an issue review and authorized incorporating
its corrections into this detailed design and ADR. The scheduling, acceptance,
and UI details below are proposed resolutions of that review, rather than claims
that each detail was individually selected or already implemented.

## 2. Scope and user-facing behavior

Each waiting entry has a target. Main agent remains the default. A named target
refers to the stable database identity of an enabled agent definition; its name
is a display label. The same target control is available before enqueueing and
in the manager for an entry that has not been claimed. Selecting before enqueue
avoids a race in which the main agent starts the task before the user opens the
manager. After successful enqueue, the next draft defaults to Main agent.

Named-target submission is available only behind an accepted turn or an existing
queue, matching the current queue's admission scope. If that scope disappears
between selection and submission, keep the draft and target with an explanation.
Never silently reroute it to Main agent. Starting a named agent from an otherwise
idle conversation is outside this version.

The first queued dispatch waits for the current main turn to complete
successfully. Thereafter, consecutive named-agent entries may launch in FIFO
order as capacity permits. A Main-agent entry starts when it reaches the head
and primary capacity is available. Only one primary turn can run in a session.

FIFO orders admission, not completion or dependencies. For example:

1. Research A is assigned to a named agent and launches.
2. Research B is assigned to a named agent and launches.
3. A Main-agent prompt saying "summarize the findings" may start before either
   child finishes. It has no implicit wait for A or B.

The manager explains: "Background tasks may still be running when the next
prompt starts." Users can inspect the fleet or wait for its completion follow-up.
Dependency barriers and automatic dependency inference are outside this design.
Thread scheduling and the order of physical provider requests are unconstrained;
the coordinator guarantees the order in which requests are accepted for launch.

The ten-entry and text-size limits remain. Waiting and starting entries count
toward the queue limit; accepted children count toward fleet/runtime limits.
Recognized slash commands retain their current immediate execution/refusal rules.
The target control applies to queued text, not to the slash-command dispatcher.

## 3. Architecture and ownership

| Owner | Responsibility |
| --- | --- |
| Queue registry | Immutable text/target entries, revisions, FIFO, claims, pause and visible wait state. Owner-thread confined. |
| Queue coordinator | Admission, one active drain per session, dispatch claims, acceptance reconciliation, primary-slot reservations, explicit wake yield, stop/close ordering. |
| Console controller | Owning-session context, readiness and project-consent checks, source conversation/path identity, transcript projection. |
| Agent bridge and shared child-launch service | Resolve child configuration, acquire execution ownership, commit accepted launches, start/retain workers and clients, publish settlement. |
| Fleet coordinator | Child handles, conversation cap, steering, cancellation, status and retained continuation state. |
| Runtime capacity owner | Actual occupied child/tool/model resources across conversations, including cleanup after logical cancellation. |
| AgentRunsDB and its existing automatic-work ledger | Accepted launch identity, run/chain provenance, durable result claims, automatic allowances and restart fencing. |
| Wake coordinator | Saved completion intake, finite same-chain batches, primary admission, acceptance and delivery recovery. |

Extract the existing child configuration and launch path so model-requested
spawns, continuations, and direct queue launches use shared execution machinery.
Keep the model-spawn and skill-specific policy checks at their respective entry
points. A direct launch has its own trusted queue authorization, rather than
pretending to be a model tool call or creating a synthetic primary run.

The bridge must retain each accepted child's service, cancellation event,
approval revocation capability, provider client/lifeline, run log and change-review
scope until their real owners settle. Registering only a FleetHandle is
insufficient. Queue completion and primary-turn teardown cannot release these
resources. No provider, database, join, or user callback runs under a fleet or
capacity lock.

The current working tree contains implementation work for ADR-134/135, including
runtime leases and durable automatic acceptance. The implementation task for
this feature must verify those dependencies on its actual branch. This document
does not treat uncommitted work in another area as merged or independently
verified evidence.

## 4. Entry and launch identity

Extend the immutable queue entry and its body-free render projection with:

- A typed target: Main, or Named agent with `definition_id` and display name.
- A wait reason when applicable: child capacity, primary capacity, configuration,
  context review, or uncertain launch. Waiting is distinct from manual pause.
- A claim/attempt identity scoped to the entry, queue owner and runtime, carried
  privately to dispatch; UI snapshots expose only ordinary status and IDs.

Target edits use the same entry identity and optimistic revision checks as text
edits and reordering. A claimed entry is locked. A capacity-waiting entry has not
been accepted and remains editable, reorderable and removable. Such mutations
invalidate any previously prepared context and schedule a fresh head check.

Resolve a definition once per launch attempt using its stable ID and current
enabled state. Freeze that definition/configuration for the accepted child and
record its fingerprint. A rename preserves identity. Deletion and recreation
under the same name do not retarget an existing entry. A missing/disabled
definition pauses dispatch with the entry intact; the user can retarget or remove
it. Definition edits affect attempts prepared afterward, not an accepted child.

Accepted delegated work requires explicit durable provenance. Extend run storage
with the equivalent of `launch_origin=queued_user`, a unique accepted queue-entry
identity, runtime owner identity, stable definition identity, and a source
transcript-owner/branch anchor. Existing model-spawned rows keep their current
parent relationship. Queued-user runs have no invented model parent.

Each accepted queued-user request creates a new immutable work-chain identity,
as an ordinary accepted queued Main-agent request already does. Its initial
child has manual origin and a contained child budget with depth one. Wake turns
and their descendants use that chain's automatic allowance. A later user request
never reparents old children or replenishes their automatic allowance.

Use the accepted run record as the launch receipt; add no durable waiting-task
table or general job scheduler. Add a uniqueness constraint for accepted
queue-entry identities and scope-checked lookup. Queue IDs alone grant no
execution permission. Number the schema migration from the implementation
branch's actual schema head; do not pin the version observed while designing.

## 5. Acceptance and failure boundaries

The shared launch operation returns a typed outcome: `accepted(receipt)`,
`already_accepted(receipt)`, `capacity_wait(reason)`, `refused(reason)`, or
`uncertain(identity)`. Only the first acceptance obtains execution authority.
The receipt contains run, entry, conversation, chain and runtime identities,
without task bodies. Rediscovering a receipt permits bookkeeping, not execution.

The required order is:

1. Claim the current head and capture its revision, owning session, source path,
   target and lifecycle generation. Resolve readiness and configuration without
   a model call. Perform the existing project/provider consent checks.
2. Acquire the shared physical child lease and conversation fleet reservation.
   If either is unavailable, release any partial acquisition, return the entry
   to the head and enter visible capacity waiting. No accepted row is created.
3. After every asynchronous boundary, recheck the claim, source context,
   session/queue lifecycle, runtime fence and target validity. Marshal queue
   operations to its owning thread; do not inspect mutable queue state from a
   child thread.
4. Commit the work-chain and accepted run/entry association together in
   AgentRunsDB, under the durable synchronization policy already used for
   execution fences. Preallocate the transcript-owner identity. This is the
   acceptance boundary; no provider call or tool execution may precede it.
5. Publish the receipt to the coordinator, transfer the claim out of queue
   accounting, and project the accepted request into the transcript. Launch the
   child through shared execution machinery using the precreated run identity.
   UI publication is best-effort; it cannot grant or revoke execution authority.
6. A child launch worker rechecks runtime revocation/cancellation immediately
   before actual execution. A concurrent Stop/close invalidates new dispatch
   even if it arrived just after acceptance. Settle accepted-but-unstarted work
   as cancelled/error as appropriate, retaining its receipt.
7. Continue draining after acceptance. Result collection belongs to the fleet.

Commit grants permission to attempt starting the child, not a promise the task
will succeed. A thread-start failure after commit is an accepted failed child:
record the failure, release physical reservations that never started, and expose
it through the fleet. Do not put it back into an automatically draining queue.
Only a proven pre-acceptance refusal is eligible for the same entry's retry.

If the start succeeds but a receipt callback, transcript write, or queue update
fails, reconcile by accepted entry identity. Never call the launch again to
repair UI. If persistence/ownership is ambiguous, stop that queue in a review
state; absence of a visible message is not evidence that execution did not start.

A capacity retry never consumes a spawn allowance or automatic allowance. Direct
requests are not charged to the previous primary's per-turn spawn counter.
Existing model/skill spawns retain their shared per-turn limit. Accepted direct
children use the same physical caps, tools, bounded child duration and depth-one
configuration as other fleet children.

For an ephemeral session, the same receipt and one-owner execution fence live in
the existing ephemeral runtime/store instead of creating persistent records.
Idempotence is process-local and ends with that runtime; no durable recovery or
replay is promised. All launch paths must honor the session's existing feature
restrictions before reaching either acceptance store.

## 6. Queue scheduling and physical capacity

The queue coordinator remains the sole admission owner for its entries. It never
skips a capacity-blocked head. Capacity notifications are hints to retry atomic
admission, not transferable permits or proof that space is still free.

Keep the existing primary reservation across immediately runnable entries. When
the head is waiting for child capacity, release the primary reservation because
the queue has no primary work executing. Launching further child entries after
capacity opens does not need a primary slot. When a Main-agent entry reaches the
head, reacquire primary capacity; if unavailable, show "Waiting for main-agent
capacity" and retry on a capacity change. Never start two primaries in a session.

This explicitly extends ADR-046: automatic primary reacquisition is allowed for
an already admitted, visible queue that released its slot during delegation.
It does not create a hidden queue for an idle session's refused manual send.
Unchanged all-Main queues retain their current reservation behavior.

Read both per-conversation fleet capacity and the app-owned runtime lease pool.
A logically finished or cancelled child can still occupy a physical lease while
an abandoned worker or provider cleanup continues. Retry when actual resource
release is published, even if the resource belongs to another conversation or a
retired service. Logical settlement alone must not cause an endless retry loop.

Provide a content-free capacity-change notification from the existing capacity
owner, emitted after its lock is released, plus a monotonically changing revision
or equivalent subscribe-and-recheck contract. The queue registers its wait before
rechecking admission so a release between refusal and subscription cannot strand
it. Wake all relevant queue coordinators through the owning event loop; coalesce
notifications to one scheduled drain per session. Re-read live limits at admission.
Per-session FIFO is guaranteed; cross-session fairness beyond the existing
runtime policy is not introduced here.

Queue mutations, capacity changes, setting changes and explicit Resume are
relevant retry triggers. Manual pause, context review, missing definition,
uncertain acceptance and shutdown do not receive blind retry timers. A child
failure after acceptance does not fail an unrelated primary turn or restore its
queue entry. A Main-agent failure/Stop retains the current queue recovery rules.

## 7. Completion wakes and explicit queue yield

On durable child settlement, publish the existing individual-completion event
with explicit queued-user provenance. Such a child is eligible independently of
any parent-terminal timestamp, including one that finishes immediately after
launch. Update durable recovery discovery and result-claim validation alongside
the hot event path; changing only the event path loses results after restart.

Ordinary model children retain the existing distinction between results handled
within their spawning turn and survivor results. For queued-user children,
derive eligibility from accepted launch origin, chain and terminal result state.
Do not attach them to the last primary solely to satisfy legacy parent joins.

Preserve ADR-135's coalescing, same-chain claims, automatic budgets, manual reserve,
one primary per conversation, global wake limit and durable acceptance. Multiple
independent queued-user entries create separate chains; do not merge their
results into one automatic allowance. The main agent can explicitly inspect other
results through fleet tools under its current permissions.

While a draining or paused queue owns the next generation, automatic completion
wakes remain pending. Show "Results ready; queued prompts have priority" with the
existing fleet status. An empty queue releases ownership and allows normal wake
admission. No completion implicitly interrupts a primary or changes queue order.

Add one explicit action, "Handle ready results", for a queue with eligible saved
results and no active primary or starting claim. It:

1. Pauses queue dispatch and releases its primary reservation.
2. Selects the oldest eligible chain and its currently ready result IDs.
3. Grants the wake coordinator one process-local, revision/context-checked
   exception to queue priority for that selected batch only.
4. Leaves the queue paused after the accepted wake, whether it succeeds or fails.
   The user explicitly resumes remaining queued work.

The exception must pass both the wake scheduler's priority gate and the
controller's acceptance gate. It bypasses queue ownership only; ordinary draft
priority, approvals, source-path checks, primary capacity, autowake settings and
automatic budgets still apply. Report the precise reason if one of those blocks
delivery. This action does not constitute a new user task or reset chain limits.
At a primary cap of one, the current automatic manual-reserve policy still
prevents the wake; preserve the results and explain why.

Consume the exception at acceptance. While awaiting capacity, a Resume, queue
mutation, source-context change, stop, close or runtime replacement invalidates
it. A pre-acceptance refusal ends that request to yield, with results intact.
Later results are excluded from the selected batch. A duplicate callback cannot
authorize another wake. Any accepted attempt's bookkeeping/recovery remains
owned by the durable wake ledger.

Task-only context does not eliminate branch ownership. A delegated request and
its eventual wake remain anchored to their accepted source path. If that owner
is absent from the active path, save the result for review instead of delivering
into a different branch. Ordinary linear appends remain compatible. Source-path
deletion or supersession must not silently reparent the run. Explicitly applying
an old result to another path is new manual work, outside automatic delivery.

## 8. Context, authority and supported configuration

Resolve one immutable execution snapshot from the entry's owning session, never
the visible tab or a previous primary's mutable service fields. It includes
provider/model resolution, capabilities, generation settings, selected workspace
binding and locator identity, project-instruction configuration and base tool
catalog policy. Revalidate the existing live authority and credentials at the
normal execution boundaries; never cache approval decisions inside the queue.

The child receives canonical task text, the fixed child system prompt plus the
selected definition's appended instructions, tool schemas/environment note, and
applicable ephemeral project instructions. It receives no inherited conversation
messages, summaries, private provider continuation, attachments, staged evidence,
persona/world-info history, or automatic RAG payload derived for a main turn.
The task-only label describes message inheritance, not a promise that permitted
tools cannot read authorized workspace resources. Tool restrictions still apply.

Use existing explicit text/skill-reference validation without introducing a
main-model helper or silently expanding a reference into conversation context.
If a queued task depends on a skill execution path that ordinary depth-one
children cannot use, return an actionable pre-acceptance refusal. Do not route
that task through the main agent or give the child spawn/skill execution powers.

The selected definition only narrows the owning session's tool authority.
Read-only bindings remain read-only. Startup and nested project instructions use
ADR-069's selected-binding/preflight boundaries and remain ephemeral. All child
tool approvals retain their own run identity and existing per-call review.
Project/provider consent required for this execution remains effective when the
session is not currently visible.

MVP readiness requires native agent execution and background fleets enabled,
including compatible `max_live_subagents` and `subagents_outlive_turn` settings.
When settings disable that path, refuse before acceptance with the draft/entry
intact. Do not reinterpret a cap of one as a threaded fleet, run the child inline,
switch provider, or enable a disabled setting automatically. Definition model
overrides follow the existing provider/endpoint boundary.

Retain the existing admission refusal for staged attachments/evidence. A direct
dispatch must never consume composer state, one-shot prefill or staged riders.
For this MVP, a rider staged while waiting also pauses dispatch under ADR-046,
even though the direct child would not consume it; no second staging contract is
introduced. Task-only mode does not bypass existing context-epoch review.

## 9. Transcript, history, usage and review

Before acceptance, queued task text stays in the memory-only queue and explicit
queue-editor views under current privacy rules. After acceptance, normal durable
run storage can retain the task and execution details. Do not copy raw task text,
automatic instruction bodies or credentials into new telemetry or lifecycle logs.

Show an accepted user-request record labeled "Delegated to <agent>" with its run
link. It uses stable message/run identity and carries explicit delegation origin.
It is a transcript record, not an assistant response. Main-agent provider-history
construction must exclude it as an outstanding user prompt so a later main turn
does not execute the same request again. The completion notice supplies the
bounded task/result attribution when the wake is admitted. Main-targeted entries
retain their normal user/assistant transcript behavior.

Allocate that owner-message identity before committing acceptance, store it with
the source path in the run receipt, and append its transcript projection
idempotently after commit. The two databases do not share a transaction. If
projection fails, recover by receipt identity; never relaunch to recreate a
message. Missing/unverifiable source anchors prevent automatic delivery until
repaired against the recorded source path or manually reviewed.

History restoration, prompt export and provider-payload builders must preserve
the distinction between a delegated request record and a normal user turn.
Retain ordinary user-authored export semantics; metadata projection is not a
second prompt execution path. Temporary sessions use their existing ephemeral
storage policy and in-process acceptance identity; they do not acquire durable
queue, run-log, transcript or result storage just to use delegation. Runtime
replacement cannot restore or auto-replay their work.

Once accepted, the task moves from queue accounting to the fleet. An empty queue
does not imply all work is done. Keep the fleet running/approval indicators and
cross-session attention visible. Child usage belongs to that child's accepted
run/request, not the previous main reply. Reuse ADR-131's durable run accounting
and existing billing separation. Never synthesize primary tokens or double-count
the child when a wake later reports its result.

The bridge's change-review scopes must cover a child launched without a primary.
Concurrent main/child file changes retain the existing shared-workspace
attribution limits; delegation does not allocate a worktree or isolate file
writes. Preserve aggregate/concurrent labels rather than claiming a diff belongs
exclusively to one agent when the evidence cannot establish that.

## 10. Pause, cancellation and lifecycle

| Action/event | Queue behavior | Accepted children |
| --- | --- | --- |
| Pause while only dispatching/waiting | Pause before the next claim; release idle primary reservation. | Continue. |
| Pause during a Main-agent turn | Existing pause-after-turn behavior. | Continue. |
| Remove/Clear | Remove waiting entries only; claimed entries stay locked until settled. | Unaffected. |
| Stop Main agent | Existing Stop plus queue pause. | Existing fleet stop policy applies. |
| Stop one child | Affect only that child; other entries retain their queue policy. | Revoke that child's outstanding approvals and request cancellation. |
| Stop all sub-agents | Pause dispatch and invalidate starting/yield authorizations before cancellation. | Cancel all children in that conversation. |
| Confirmed session close | Tombstone queue before cancelling work and discarding unsent entries. | Cancel and retain actual resource ownership until cleanup finishes. |
| Leave Console | Existing count-aware queue discard and fleet survivor policy. | Accepted eligible survivors remain owned by the runtime. |
| Quit/process loss | Unsent memory-only entries are not restored. | Persistent accepted runs/results use existing interrupted-work recovery; no automatic relaunch. |

Stop all must pause even during a gap with zero live children but a delegated
entry waiting or starting. Child admission and cancel-all use the same lifecycle
generation so a raced acceptance is either prevented or included in cancellation.
Cancellation may not immediately release physical capacity. A later capacity
notification cannot override pause, closing, or shutdown.

A successfully accepted child that fails reports a fleet failure; its request
is not automatically retried by the queue. Explicit cancellation is reported as
a user decision and never causes the queue to recreate that task. Existing
manual fleet continuation/retry can create new work under fresh authority.

Persistent results remain discoverable without the original mounted screen or
attention badge. Active/uncertain work from a revoked runtime requires review,
following ADR-135. A completed result can enter ordinary pending delivery after
recovery proves its eligibility. Neither case restores the old unsent queue.
Keep the existing distinction between screen attachment and process restart.

## 11. Verification requirements for implementation

These are acceptance requirements, not claims of tests already run for this
feature. Use targeted registry, service/controller, real-SQLite and mounted
production-hierarchy UI tests. Use gated provider/tool doubles to prove ordering
and actual dispatch counts, then isolated live TUI verification for the visible
flows. A full-suite run requires the repository's explicit opt-in.

| Case | Required observable outcome |
| --- | --- |
| All entries target Main agent | Existing sequential acceptance, pause/retry and composer behavior remain intact. |
| Two delegated entries followed by Main | Launch acceptance preserves FIFO; Main can start while children are gated; no main-model call is made to launch a child. |
| Current main turn fails or is stopped | No queued entry starts until the existing recovery action permits it. |
| Enqueue races with the original turn ending | A named target is preserved or refused intact; never silently sent to Main. |
| Target edit races with claim | One revision wins; no mismatched text/agent pair or duplicate launch. |
| Definition rename, disable, delete/recreate | Stable ID governs resolution; disabled/missing target pauses without substitution. |
| Local or runtime child cap full | Head remains editable and later entries do not pass; no row/thread/allowance consumption. |
| Cancelled worker still cleaning up | No new physical admission until real release, even if its fleet row is terminal. |
| Release between refusal and wait registration | Subscribe/recheck observes release; queue cannot remain stranded. |
| Release in another conversation or retired owner | Waiting queue is retried on its event loop. |
| Primary capacity consumed while child wait released reservation | Child may launch; a later Main entry visibly waits and reacquires without bypassing FIFO. |
| Two simultaneous capacity notifications | At most one drain/claim and one launch per entry. |
| DB acceptance commit fails | No model/tool execution and no UI acceptance substitute. |
| Start fails after accepted commit | Exactly one failed accepted run; resources unwind; no automatic queue retry. |
| Child finishes before receipt/UI callback | Receipt reconciles once; completion remains eligible despite having no model parent. |
| Callback/transcript write fails after launch | Request is recovered by run identity without another execution. |
| Queue empties but child runs or needs approval | Queue count is zero; fleet still shows running/waiting work with functioning controls. |
| Child completes behind draining/paused queue | Saved result and pending reason are visible; no implicit queue bypass. |
| Handle ready results | Exact one-chain batch can pass queue gates once, retains automatic origin/budget, and leaves queue paused. |
| Yield races with Resume/edit/Stop/close/context change | Stale grant cannot pass either scheduler or acceptance gate. |
| New result arrives during yielded wake | It remains pending; it is not accidentally marked delivered with the selected batch. |
| Background session/provider/workspace changes | The owning-session snapshot and fresh project/permission checks govern execution. |
| Task-only provider payload capture | No history, summary, continuation or staged rider is inherited; applicable ephemeral guidance remains correctly scoped. |
| Delegated request restored/exported and a Main turn starts | User provenance is retained, but task text is not replayed as a pending Main-agent request. |
| Branch changed after launch | Off-path result remains saved for review, with no automatic delivery into the new branch. |
| Stop all races with a waiting/starting entry | Queue pauses first; no replacement child starts after cancellation. |
| Reopen/remount/restart, with lost badges | Persistent accepted results remain discoverable; uncertain execution is never replayed; unsent queue is absent. |
| Ephemeral conversation | Acceptance remains idempotent in process and produces no newly durable user/task state. |
| Usage and change-review | Direct run stays inspectable without a primary parent; no duplicate billing or false exclusive change attribution. |

## 12. Delivery boundaries and review record

Implementation planning should sequence independently verifiable changes around
accepted-run provenance/recovery, reusable child execution, queue admission and
capacity waits, and the target/result-control UI. Each implementation task must
read its Backlog record and link ADR-137. Keep the feature unavailable until its
complete acceptance-to-result path is integrated; an exposed target picker with
no working recovery/cancellation path is not a shippable slice.

No cross-provider agent routing, shared transcript fork, durable waiting queue,
dependency graph, nested delegation, automatic executor selection, new workspace
isolation, or separate worker subsystem is included. A dependency on the recently
implemented fleet contracts is explicit and must be rechecked before coding.

The design review identified seven gaps. Their resolutions are launch/chain
provenance (sections 4 and 7), queue/wake arbitration (6 and 7), pre-enqueue target
selection and stable definition identity (2 and 4), acceptance receipts (5),
physical capacity waiting (6), session authority (8), and stop ordering (10).
Additional review covers provider-history exclusion, source-path recovery,
temporary conversations, usage and change-review ownership (7 through 11).

The current task ends with a proposed, self-reviewed design and ADR. Writing an
implementation plan or changing runtime code requires a subsequent implementation
planning request or approval of that next stage; the documentation status must
not imply that delegation has shipped.
