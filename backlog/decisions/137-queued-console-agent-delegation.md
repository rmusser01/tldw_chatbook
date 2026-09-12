# ADR-137: Direct delegation of queued Console prompts

- Status: Proposed; detailed design ready for review.
- Date: 2026-09-08
- Design task: [TASK-32050](../tasks/task-32050%20-%20Specify-direct-delegation-of-queued-Console-prompts.md)
- Spec: [Queued agent delegation](../../Docs/superpowers/specs/2026-09-08-task-32050-queued-agent-delegation-design.md)
- Proposed amendments: [ADR-046](046-visible-bounded-console-prompt-queue.md) for mixed-target admission/reservations, [ADR-134](134-fleet-admission-and-automatic-work-budgets.md) for direct user-child provenance, and [ADR-135](135-fleet-completion-delivery-and-crash-recovery.md) for parentless completion eligibility and explicit queue yield.
- Preserves: [ADR-069](069-console-project-instruction-local-state-and-preflight.md) authority/context boundary and [ADR-131](131-durable-agent-budget-accounting.md) accounting separation.

## Context

The current prompt queue admits text behind an accepted main turn and drains
separate Main-agent turns in FIFO order. Native fleet children are created from
an active model run and inherit its parent and work-chain identity. The user
wants to select a named agent for a queued request and have that agent execute
the task directly in the background. They selected launch-and-continue behavior,
task-only conversational context, supervisor wake on completion, and waiting
at the queue head when child capacity is unavailable.

Source review found that a new launch entry point alone is insufficient:

- Durable pending-result queries and chain validation currently depend on parent
  runs. A child without a parent is not automatically covered by those queries.
- Queue ownership blocks completion wakes even when the queue is paused.
- Selecting a target only after enqueueing can race with Main-agent dispatch.
- Thread launch, run persistence and UI callbacks have different failure points.
- Physical resource leases may remain occupied after a logical cancellation.
- Children inherit context/authority and resource cleanup from their spawning
  service; a direct child needs those owners without a primary model call.
- Stop-all currently cancels live fleet work; automatic queue launches could
  immediately replace those children unless dispatch is paused first.

ADR-134/135 implementation is present in the reviewed working tree. The eventual
implementation must verify those dependencies on its branch rather than assuming
uncommitted work is merged. This decision records intended behavior, not runtime
verification of the proposed feature.

## Decision

### Route requests through the existing fleet

Add an explicit per-entry Main/Named-agent target, keyed by stable definition
identity. Offer selection before enqueueing as well as revision-checked edits
to waiting entries. Main remains the default; reset the next draft's target
after successful enqueue. Missing/disabled definitions refuse without consuming
the entry, and a deleted/recreated name cannot silently substitute another agent.

The queue registry/coordinator owns ordering and acceptance. A shared child-launch
service behind the existing bridge owns configuration and actual execution.
Fleet handles, approvals, cancellation, client lifelines, run logs, retained
continuations, accounting and change review retain their existing owners.
Do not ask a model to delegate, manufacture a primary run, or add another worker
subsystem.

Only an already admitted visible queue can dispatch this way. A named-target
draft racing with queue teardown is retained/refused intact; it cannot fall back
to a Main-agent send. Idle named-agent launches and slash-command changes are
outside this decision.

### Establish independent accepted-work provenance

A directly delegated user request is a child execution with explicit queued-user
launch origin and its own work-chain identity. It has no fictitious model parent.
It receives manual admission classification, a contained child budget and
depth-one restrictions. Its completion wakes and automatic descendants consume
the finite allowance of that same chain. Old survivors keep their own chains.

Persist a unique accepted queue-entry/run association, source owner/path,
definition identity/fingerprint and runtime ownership in AgentRunsDB. Create the
chain and accepted run association atomically under the existing durable fence
policy before starting model/tool work. Use the accepted run as the receipt;
waiting queue bodies remain memory-only. Allocate the actual schema version at
implementation time.

Repeated launch requests may return the existing receipt but cannot obtain
execution authority again. Proven pre-acceptance capacity/refusal releases any
partial reservations and preserves the queue entry. Failure to start after an
accepted commit becomes a failed child, not an automatically retried queue item.
Unknown ownership/outcome requires review. Repairing a callback or transcript
projection cannot rerun the task. These are acceptance/replay rules, not an
exactly-once guarantee for external tool side effects.

### Preserve FIFO while releasing unused primary capacity

The first queued entry still waits for the current Main-agent turn's successful
completion. A delegated entry advances the queue at acceptance, without waiting
for its result. Later Main-agent entries have no implicit dependency on earlier
children. FIFO applies to admission only.

When the head is blocked on child capacity, keep it at the head and release the
unused primary reservation. Observe both the conversation cap and actual shared
runtime leases. A relevant physical release triggers atomic re-admission through
the queue's event loop, including releases in other conversations. Subscribe and
recheck to avoid losing a release event; coalesce to one drain per session.

After a delegation wait, a later Main-agent entry automatically reacquires a
primary slot or remains visibly waiting for it. This amends ADR-046 only for an
already admitted queue that released its reservation during delegation. It does
not queue an idle session's otherwise-refused manual send. All-Main queues keep
their existing reservation behavior.

### Reuse bounded completion delivery with an explicit yield

Queued-user completion eligibility comes from accepted launch provenance,
source path, chain and durable terminal state, without requiring a parent
terminal timestamp. Extend both live settlement and persistent discovery/claim
validation. Preserve the existing in-turn/survivor distinction for model-spawned
children and ADR-135's same-chain batching, limits, claims and recovery.

Automatic wakes normally wait behind queued work. Show saved results and the
queue-priority reason. When no primary or starting claim is active, a user may
choose "Handle ready results". Pause the queue, release its primary reservation,
and grant one scoped exception for the currently ready batch of the oldest
eligible chain. Apply the exception at both scheduler and acceptance gates.
The queue remains paused after that attempt until explicitly resumed.

The grant bypasses queue priority only. Draft priority, approval requirements,
source-path validity, autowake configuration, manual primary reserve and finite
automatic budgets remain enforced. It creates no new work chain or allowance.
Queue mutation, Resume, context change, stop, close or runtime replacement
invalidates a pending grant. Consume it at acceptance; later results are a
different batch. Ordinary empty-queue delivery needs no explicit yield.

Results from an off-path/deleted source remain saved for review. Automatic
delivery must not attach an old request to whichever branch is currently viewed.
The accepted request's durable source owner anchors this check across restart.

### Resolve authority from the owning session

Resolve one immutable provider/model/workspace/tool snapshot from the owning
session at dispatch, including project binding identity and consent. Revalidate
live permissions and cancellation at their existing boundaries. The named
definition may narrow tool access, never widen it.

Task-only means no inherited conversation history, summary, private continuation,
staged attachments/evidence or Main-agent RAG payload. The fixed child prompt,
definition instructions, tool protocol and permitted ephemeral project guidance
still apply. It does not imply filesystem isolation from authorized resources.
Keep existing staging refusals and depth-one skill/spawn restrictions. Unsupported
background fleet configuration refuses visibly; it never falls back to inline
execution or changes settings automatically.

### Keep controls and projections honest

At acceptance, show a user-request record labeled with its delegated agent and
run link. Exclude this record from pending Main-agent provider prompts to prevent
duplicate execution. Preserve delegation provenance through history/export and
repair transcript projection by stable receipt identity. Missing source anchors
block automatic delivery until repaired/reviewed. Do not attempt a cross-database
transaction between transcript storage and AgentRunsDB.

Accepted children remain visible and cancellable after queue count reaches zero.
Usage belongs to their own runs; reporting a result does not rebill it. Child-only
work retains client/resource and change-review ownership without a primary run.
Shared-workspace changes retain existing concurrent-attribution limits.

Stop-all pauses dispatch and invalidates pending claims/yield before cancelling
children, including when no live child exists yet. Individual cancellation affects
that child only. Removal/clear affects waiting entries; pause affects future
dispatch. Close tombstones the queue first. Capacity release never overrides
pause/close. Accepted failures and explicit cancellation do not recreate queue
entries automatically.

Unsent queues keep ADR-046's memory-only lifetime. Accepted persistent work uses
existing run/result storage and conservative recovery, with no automatic replay
of uncertain execution. Ephemeral conversations use equivalent process-local
acceptance and retain their existing nonpersistent policy. Queue completion,
screen attachment and process restart remain distinct lifecycle events.

## Alternatives considered

| Alternative | Reason not selected |
| --- | --- |
| Synthetic main-agent delegation prompt | Requires a main-model call and makes the chosen executor depend on model compliance. |
| Independent queued-worker subsystem | Duplicates execution, permissions, resource lifetime, cancellation and recovery contracts. |
| Attach new work to the previous primary | Misrepresents user-request lineage, budgets and supersession. |
| Create a dummy primary row | Avoids updating parent assumptions by introducing a model run that never happened. |
| Persist the entire waiting queue | Adds restart replay and private unsent-text storage beyond the selected scope. |
| Let later entries pass a blocked head | Violates the selected FIFO behavior. |
| Hold a primary slot throughout child-capacity waits | Occupies capacity with no primary execution and delays unrelated manual/wake work. |
| Automatically let wakes interrupt the queue | Changes next-turn ownership and can surprise a user who paused queued work. |
| Resolve agent definitions solely by name | Deletion/recreation can silently route work to a different definition. |
| Rerun after missing callback or uncertain start | Can repeat tools already executed; durable acceptance is the authority. |

## Consequences and validation

The change needs explicit accepted-run provenance, schema migration, shared
launch machinery, capacity-release notification, queue/wake arbitration and
target/result controls. It is an architectural feature, not a target field alone.
An approved implementation plan must keep those boundaries coordinated.

The specification's acceptance matrix covers mixed FIFO dispatch, target/close
races, fast completion, failed acceptance/start, duplicate callbacks, physical
cleanup, cross-conversation release, wake grants, branch changes, permission
isolation, ephemeral sessions, usage and restart recovery. Tests must observe
actual child/main dispatch counts and durable state, followed by isolated live
TUI verification. Documentation-only review is not execution evidence.

This proposed ADR does not yet supersede or edit the accepted decisions it
amends. After user review, update their metadata/references as appropriate while
preserving their historical decision text. Recheck task/ADR number uniqueness on
the implementation branch before merge.
