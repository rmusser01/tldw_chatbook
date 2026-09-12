# ADR-134: Fleet admission and automatic work budgets

Delivery/recovery amendment: [ADR-135](135-fleet-completion-delivery-and-crash-recovery.md)
replaces the proposed drain barrier and one-global-wake policy with incremental
notifications and up to two wake slots, preserving the manual reserve. Its
attempt/acceptance contract governs TASK-32036/32037; the durable ledger and runtime enforcement are implemented.

Status: Accepted
Date: 2026-09-08
Implementation status: TASK-32034/32035 implement operation ownership, tool-worker admission, and shared child admission; TASK-32036 implements durable chains/reservations/attempts and trusted lineage; TASK-32037 enforces automatic chains and projects saved-result pauses
Related task: TASK-32019
Related decisions: ADR-129, ADR-130, ADR-131

## Problem and evidence

Per-conversation handles and per-run budgets leave two independent gaps:
simultaneous background work can grow across conversations, and completion
wakes can repeatedly start fresh runs without another user instruction.

`Tests/Chat/test_fleet_budget_boundary_probes.py` records the original boundaries
using actual service/controller calls, gated local workers, and SQLite:

- Four conversations retain two children each after their parents finish:
  eight actual gated provider calls, with every conversation inside its cap.
- Six consecutive machine-origin wakes complete without adding a user row.
  This probe injects successive completion events at the real coordinator;
  it does not claim that a live model chose to spawn another child six times.
- A service marks a wedged child cancelled while its worker remains alive.
  Its coordinator reports zero live handles and admits replacement reservations.
- Four timed-out tool invocations leave four actual tool workers alive.

Console defaults currently allow 2,000 model turns, 25 million budget tokens,
and 24 hours per primary run. New automatic work inherits ordinary run settings.
These are existing defaults, not appropriate automatic-chain limits.

## Decision and scope

Use conservative configurable limits by default, as selected by the user.
Apply resource admission across the app-owned Console runtime, including
headless work and its retired/closing owners. Keep per-conversation caps too.
Manual primary runs retain their existing user-configured run budgets.

Count physical ownership separately from logical run status. A cancelled row,
terminal handle, pruned record, closed tab, timed-out wait, or returned parent
is never proof that its resource is free. An operation can also finish locally
while an external server continues work; these limits bound local admission,
not remote execution after a connection is closed.

The first implementation applies to native Console agent execution and its
plain-provider wake fallback. Direct library AgentService callers receive an
explicit injectable owner; they must share one to claim a common capacity
limit. Separate processes, server-offloaded jobs, and unrelated media/eval
workers are outside this runtime's bound. Do not claim a machine-wide quota.

## Defaults

Names below are the proposed `[agents]` configuration contract. They must not
appear as effective settings in product documentation until their slice ships.

| Key | Default | Meaning |
| --- | ---: | --- |
| `max_runtime_subagents` | 6 | Occupied child execution leases across conversations. |
| `reserved_manual_subagents` | 2 | Of the six slots, automatic work may occupy at most four. |
| `max_runtime_tool_workers` | 8 | Owned active or abandoned local tool workers. |
| `reserved_manual_tool_workers` | 2 | Automatic work may occupy at most six tool slots. |
| `max_autowake_generations` | 3 | Accepted wake turns in one causal automatic chain. |
| `max_autowake_child_launches` | 6 | Child launches from those automatic turns, including continuations. |
| `max_autowake_model_calls` | 32 | Provider calls across the automatic primary turns and their children. |
| `max_autowake_budget_tokens` | 500000 | Shared admission/accounting budget in the units below. |
| `max_autowake_output_tokens` | 8192 | Per-call maximum output, further narrowed by existing settings. |
| `max_autowake_wall_seconds` | 900 | Elapsed automatic-chain ceiling after its first accepted wake. |

Six children allow three ordinary two-child manual turns at once; two reserved
slots preserve a manual turn's normal spawn allowance. Eight tool workers bound
abandoned operations while retaining manual headroom. These are initial safety
defaults, not benchmark-derived throughput optima.

Positive limits reject booleans, non-finite values, and invalid numeric strings;
invalid configuration falls back to the default. Integer limits require an
integer. Reserved counts are clamped to `[0, total]`. Zero automatic generations
disables automatic execution but retains results. Resource limits and other
automatic ceilings do not interpret zero as unlimited. Users can raise finite
limits explicitly. Existing `autowake_enabled` remains the immediate kill switch.

## Resource ownership and admission

One runtime owner maintains an atomic, locked lease table. Acquire before a
child handle/thread or tool worker is created. Each lease has an opaque ID,
conversation/run identity, trusted origin, and a state; it contains no prompt,
task text, tool arguments, results, or credentials.

Automatic admission requires both `total_occupied < total_limit` and
`automatic_occupied < total_limit - reserved_manual`. Manual work uses the
same total limit. Acquisition never waits while holding a coordinator lock,
and no user callback, DB operation, join, or provider call runs inside the
admission lock. Refused spawns create no child row/thread and consume no
per-turn spawn allowance. Continuing a finished child acquires a fresh lease.

The root execution releases its claim after completion, but its lease stays
occupied until every owned worker and model-call lifeline has settled. In
particular, `_call_with_timeout` registers before thread start and unregisters
in the worker's own `finally`, not in the caller's timeout path. The model
lifeline releases from its actual driver shutdown/cleanup completion, including
cleanup that outlives a bounded join. Failed starts unwind exactly once.
Repeated completion or cleanup callbacks are idempotent.

Permit at most one outstanding timed-out tool worker per run. A later tool
launch from that run is refused while its prior worker still runs, preventing
one stuck run from filling the entire pool through retries. A batch that has
not timed out keeps its existing ordering; this does not introduce parallel
tool execution. A replacement manual run still observes the global tool cap.

Lowering limits changes admission immediately without cancelling admitted
work or discarding leases. Toggling threaded fleets off routes inline children
through the same admission owner; it must not create an escape from the bound.
Closing/reopening a session or replacing a bridge keeps the runtime owner until
all of its work has settled. App shutdown refuses new work before draining.

## Manual priority and origin

Origin is supplied by the trusted submit path, never parsed from messages or
tool arguments. Child spawn/continuation inherits its launching turn's origin.
Queued user submissions are manual; machine wakes and their children remain
automatic, including when the agent runtime switches to the plain-provider path.

Keep the current draft/queue priority probes. ADR-135 amends the original
one-global-wake rule to allow at most two automatic wakes in distinct
conversations. An automatic wake also leaves one of the
`console.max_parallel_runs` primary slots for manual work. At a primary limit of
one it remains pending until the user
handles the result manually or increases the limit. Manual priority is admission
priority, not a promise to interrupt an already-running provider call, resolve
an approval, or run two turns in the same conversation. No new preemption.

## Causal chain and durable counters

An accepted explicit user submission creates a new immutable work-chain ID.
Its initial manual run/children keep existing run budgets; automatic descendants
share the finite automatic allowance above. An automatic turn never creates a
new allowance. Attach chain identity when a run is created, including the
plain-provider wake path, and preserve it through continuation and retention.

Manual acceptance can authorize a new chain for new work. It never reassigns
older survivors or resets their chain's counters. Tab navigation, provider/model
changes, pruning, config reload, and app restart do not reset counters. Limits
are snapshotted at chain creation; live reductions and the kill switch may stop
further admission, but raising settings does not silently replenish old chains.

A completion wake selects one chain at a time, oldest pending first. Coalesce
that chain's eligible results and leave other chains pending. This avoids a
single mixed-source wake reparenting old work into a fresh allowance. Preserve
the current conversation-drain barrier and final-child billing reconciliation;
TASK-32020 separately chooses incremental delivery timing.

AgentRunsDB owns durable chain counters and uniquely identified reservation
records. Atomic transactions enforce bounds before dispatch; unique reservation
IDs make retries, late completion, and duplicate refunds idempotent. Reserve a
generation before wake scheduling and commit it at actual submission acceptance,
not when the full turn returns. Pre-acceptance refusals release only their own
uncommitted reservation. Accepted cancellation still consumes a generation.
Child launch and provider-call counters reserve before starting actual work.

Reserve estimated prepared-input tokens plus the resolved maximum output before
each automatic provider call. Use the exact request after instruction/tool
preparation. Concurrent reservations share one remaining balance; they cannot
each spend the same remainder. Successful outcomes replace their reservation
with the existing budget counter; missing/uncertain outcomes conservatively
retain the reservation and mark accounting uncertain. Never turn unknown usage
into zero. Actual usage above an estimate can cross the ceiling: stop further
admission and show the overage. This is a budget-token admission policy, not a
hard raw-token or currency guarantee; ADR-131 billing separation remains intact.

Elapsed time includes queueing and human approval waits once automatic work has
started; it never resets per wake. Check it before each admission and through
the existing cooperative cancellation path during execution. A deadline cannot
kill Python threads or prove a remote call stopped; their leases stay occupied.
Persist the start/deadline for restart recovery, use monotonic elapsed time
within a process, and require review after a backward clock anomaly rather than
granting additional unattended time.

## Exhaustion, crash recovery, and visibility

Capacity refusal is retryable after actual release. Present a reason such as
“Background capacity is occupied; 2 workers are still stopping.” Never describe
those slots as available because their run rows say cancelled. Keep drafts and
completed results, and expose manual cancellation through existing controls.

Generation/call/token/time exhaustion pauses that chain's automatic work and
preserves the pending completion ledger and unseen indicators. Show “Automatic
work paused: budget reached. Results are saved; send a message to continue.”
The next manual instruction creates new work explicitly; merely viewing results
or approving a tool does not replenish a budget. Paused work gets no retry timer
until a relevant capacity change or explicit user action occurs.

An interrupted reservation is not automatically refunded after restart. A wake
whose acceptance is uncertain becomes review-required, keeps its generation
charged, and is not replayed automatically. Legacy runs with no chain identity
remain inspectable but cannot acquire fresh automatic budgets on restart.
The exact delivery-attempt/crash-state schema must be settled under TASK-32021
before durable wake enforcement ships. No exactly-once side-effect claim is made.

Expose total/automatic occupancy, stopping workers, paused chains, and exhaustion
reason through typed runtime snapshots to the existing fleet surface. These are
metadata only. Do not log message bodies or feed policy reservations into billed
usage. A modal, view refresh, or notification failure cannot change admission.

## Alternatives and rollout

- Per-conversation counters alone fail the eight-child probe; counting terminal
  handles fails the worker-exit probe. A semaphore released on timeout has the
  same defect. Use owner-held leases.
- Opt-in limits preserve behavior but leave unattended defaults unbounded. The
  user selected conservative defaults instead.
- Disabling spawning on every wake is simpler but prevents useful follow-up
  work. A finite shared chain allowance supports bounded follow-up.
- A process-global singleton hides ownership and contaminates separate runtimes.
  Inject the app-owned object and retain it through cleanup instead.
- A complete scheduler with preemption, fair queues, and currency pricing would
  mix separate policies. Keep immediate admission/refusal and budget tokens.

Ship independently verified slices: worker ownership/tool capacity, child
admission/manual reserves, durable chain reservations, then wake enforcement and
visible pause/recovery. Document each as implemented only after its actual
runtime path and cancellation/restart tests pass. Until then the baseline
limitations above remain real.
