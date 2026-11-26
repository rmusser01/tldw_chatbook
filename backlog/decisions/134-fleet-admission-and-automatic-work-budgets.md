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


## ADR-281 review-boundary amendment (TASK-35582, 2026-10-26)

[ADR-281](281-primary-agent-multi-model-code-review.md#native-acceptance-and-continuation-implementation-amendment-task35582-slice-7)
adds host-sealed exact-next-review-slot eligibility to the existing
`ConsoleFleetWakeCoordinator` / automatic-work ledger. AgentRuns' normal 24→25
migration adds `automatic_review_boundary_claims` and nullable boundary identity
on wake attempts; no source or report bodies enter that database. The existing
two-child turn ceiling, finite original chain, priority, acceptance, prepared-abort
and consumed/uncertain replay fences remain authoritative. Restart cannot restore
an ephemeral review bundle or mint a new stage grant. The original decision above
retains its scope; the [qualification record](../../Docs/superpowers/qa/2026-10-18-multi-model-code-review/README.md)
distinguishes deterministic checks from missing live-provider evidence.

## Retained unproven operations amendment (TASK-40716, 2026-11-13)

A tool operation whose cleanup is unproven is a retained operation. It stays stopping and keeps counting toward `max_tool_workers`, so "N workers are still stopping" stays true. It ends its run instead of refusing the run's later calls one by one. ADR-155 and ADR-356 §6 fit as they are: the drain stays unproven, and an unproven surviving worker keeps its real resource claims.

**Owner decision, 2026-11-13 ([TASK-40716](../tasks/task-40716%20-%20An-unproven-tool-cleanup-parks-a-worker-forever-refusing-later-tool-calls-and-holding-a-tool-slot.md)).**
- When a tool operation's cleanup cannot be proven, its run ends at once with a reason that names the unproven cleanup. This applies to every run, not only goal runs. The resource stays held and is never reused.
- The run has already ended by then. An explicit user abandon releases only the tool-worker slot. The resource, its storage leases and any activity claim stay held until restart (ADR-155).
- The abandon action sits on the run's terminal card: the run's reply, which the explanation naming the unproven cleanup follows.
- At app exit, disposal records retained items and stops waiting for them.

**Implementation note (not an owner decision).**
- `OwnedOperation.retain_unproven` records the retention on the runtime's capacity.
- A worker that still holds the original scopes waits on that record on a daemon thread until restart (no proof path exists yet; `UnprovenRetention.prove` would end the wait early, but no production code calls it). `UnprovenRetention.hold` never unwinds custody it has entered: on a joined thread it logs an error and still holds.
- Raw CLI user commands, Undo, @git reference expansion and the Console's startup SSH instruction read run their custody on a daemon worker, so no joined pool thread ever holds. Undo also marks its owner unproven.
- In an ordinary Console turn, @git references and the startup SSH instruction read borrow the turn's owner, so the turn's next tool reservation is refused as cleanup-unproven. That ends the agent run at its first tool call ("the cleanup of native Git could not be proven"), or, when the turn reserves a tool slot while composing its agent request (MCP), stops the turn before the run starts.
- A retention that arrives after a timeout, cancel or per-call abandon ends the run at its next tool call.
- A custody-bearing (SSH or Raw CLI) call never holds a run thread: with no timeout it still runs on a daemon tool worker, and a definitive-after-start one is refused before it starts.
- A workflow run held only by unproven cleanups (each of its tool workers is retained or has returned) does not stop exit. `ConsoleRuntime.dispose` closes every other owner and then reports `workflow_original_shutdown_unresolved`, and the app's exit runs its remaining owners before re-raising, logging any further failure it does not raise. A cancellation requested meanwhile is still re-raised, chained to the retained runs. A run whose custody is still live keeps stopping disposal before capacity closes.
- A goal iteration that ends on it is uncertain, so it needs recovery.
- The Release row follows the run's retained records, not how the run ended, and releases every slot the run still holds. A closed and reopened conversation finds it again by the card's persisted id.

## Release outside a run's card amendment (TASK-41307, 2026-11-17)

This extends the TASK-40716 decision that the abandon action sits on the run's terminal card. The card keeps it; it is no longer the only place.

**Owner decision, 2026-11-17 ([TASK-41307](../tasks/task-41307%20-%20Unproven-cleanup-records-outside-an-agent-runs-reply-have-no-Release-button-so-their-tool-slot-stays-held-until-restart.md)).**
- One Console-wide "Held tool slots" list releases every retained record. It sits behind a status-line segment that shows only while an unreleased record exists, and it includes run records that also have a card, so it shows every held slot. Sub-agent records are released only there.
- A turn that stops before its run exists shows the cleanup-unproven explanation naming the tool instead of the "Response recovery failed" prompt, with Release directly under it.
- A Raw CLI command's "Cleanup: Unproven" row carries Release too.
- Release still returns only the tool-worker slot. The resource, its storage leases and any activity claim stay held until restart.

**Implementation note (not an owner decision).**
- A record is keyed by its run, else by its owner's execution. Undo, Change Review jobs with their own owner, an @git reference outside an ordinary turn and a pre-run stop are never bound to a run; a Change Review snapshot job that borrows an ordinary turn's owner carries that turn's run id. A transcript row can carry several keys, so a retried reply keeps its pre-run record's Release.
- The list reads the runtime's capacity, not the agent bridge's, which is allocated only once an agent turn runs; a session that only ran Raw CLI or Undo still lists its records.
- `CleanupUnprovenRefused` carries its retention, so the accepted-turn failure handler and the Raw CLI command can point Release at it. A pre-run stop keeps the accepted turn's dispatch recovery; only its copy changes, and it is set just before that recovery is restored, so an early return leaves none behind.
- The segment refreshes on the Console's sync, after a card's Release and when the list closes. A Change Review job retained in the background between syncs shows at the next one.

## Change Review initialization custody amendment (TASK-41635, 2026-11-18)

Implementation note, not an owner decision. It extends the TASK-40716 note that custody runs on a daemon worker "so no joined pool thread ever holds" to Change Review filesystem initialization, which that list missed. It also says what a retained cleanup does to initializations that come after it.

- Initialization keeps its fixed two-thread pool. Each native job runs on a daemon custody worker (`run_on_custody_thread`) while its pool thread waits. When the job's Git cleanup is retained as unproven, the pool thread records that root and takes the next job, and the custody worker keeps the original scopes until restart. Before, the pool thread itself held. Two retained cleanups then left every later job queued, reading "Preparing", with nothing saying why.
- This does not let later roots prepare while the hold lasts. A retained cleanup's activity claim stays held until restart (the TASK-40716 decision), and initialization either borrows an opaque turn's owner or takes its own opaque claim. Inside the same turn, the owner's tool reservation raises `CleanupUnprovenRefused` until Release frees the retained operation; after that, a still-live turn admits initializations again. Anywhere else, the held claim refuses a later initialization until restart. `_InitializationSource.admit` then raises `CleanupUnprovenRefused` when the refusing claim belongs to a retained cleanup's owner. That cleanup can be any retained kind (native Git, Raw CLI, Undo, an agent's tool), so the copy does not name Git.
- Such a root, the retained one included, reads the `RESTART_REQUIRED` readiness, which says that change history waits for an app restart because an earlier cleanup could not be proven. Settings ▸ Workspaces shows it on its own line with no retry, and "Retry failed preparation" skips it. `ChangeReviewConsentService._fail` maps each refusal to its readiness in one place; "busy; retry from Settings" is left for refusals a retry can clear.
- Threads stay bounded: the two pool threads, at most one live custody worker per pool thread, and one per retained cleanup. Until Release, the retained owner refuses further reservations (`CleanupUnprovenRefused`) and its claim refuses every other initialization, so only jobs admitted before the first retention can still be retained, at most one per folder admitted with it. A Release frees the retained operation, so a still-live turn can then admit, and possibly retain, more initializations: each further custody worker follows a user's Release. The custody worker itself lives until restart.

## Retained Change Review initialization claim amendment (TASK-41950, 2026-11-25)

**Owner decisions, 2026-11-25 ([TASK-41950](../tasks/task-41950%20-%20A-retained-Change-Review-initialization-jobs-opaque-claim-refuses-every-later-workspace-ac.md)).** The first applies the TASK-40716 decision to a Change Review initialization job that runs under its own owner (outside any ordinary turn). The second was made after the first showed that such a claim refuses every chat's Send until restart.
- When such a job's Git cleanup is retained as unproven, the job keeps its opaque workspace activity claim until restart, as every retained operation keeps its claim.
- Releasing the job's tool slot (Release in "Held tool slots", TASK-41307) frees only the slot. The claim stays held.
- A Send refused by a conflicting claim that is held until restart says so: "Workspace activity is held by a cleanup that could not be proven; restart Chatbook to clear it." A tool the model calls whose widening is refused the same way says "<tool> is unavailable: workspace activity is held by a cleanup that could not be proven; only restarting Chatbook clears it. Continue without it." Ordinary conflicts, which do finish, keep "Workspace activity is busy; retry after it finishes." and "<tool> is unavailable while other workspace activity is running: … retry after that activity finishes."

**What this costs, accepted.** An opaque claim conflicts with every other claim, and the Console runtime keeps one activity registry for all chats. So until restart every later Console Send is refused at ordinary-turn admission, in any workspace and not only the job's. Later initializations read `RESTART_REQUIRED` (section above).

**Implementation note (not an owner decision).**
- `WorktreeActivity.restart_refusal(footprint)` is the "held until restart" predicate. It returns a `CleanupUnprovenRefused` (no retained record) when a claim that conflicts with `footprint` has a member whose owner's cleanup is unproven (`ExecutionOwner.cleanup_unproven`). Such a member's drain can never be proven, so `_retire_if_settled` never retires the claim. A member drained unproven always has that flag set, so the flag alone decides.
- It does not cover every claim held until restart. `_retire_if_settled` also needs each member's durable settlement, and two paths drain a member proven but never settle it durable:
  - a Change Review snapshot whose write failed outside a workflow wave (`_SnapshotTicket.returned`);
  - an ordinary turn whose run-row check failed (`OrdinaryTurn.host_returned`).

  Those claims still get the busy copy. The registry cannot tell "not yet durable" from "never", so they are [TASK-42605](../tasks/task-42605%20-%20Claims-held-until-restart-by-a-non-durable-settlement-still-say-retry-and-a-queued-snapsho.md), together with a Change Review snapshot that queues behind a held claim.
- The predicate covers a retained operation's owner: an initialization job, an agent turn's own tool (TASK-40716), or a sub-agent that joined the turn's claim. It also covers an owner marked unproven with no retained record (`mark_cleanup_unproven`, for example an agent worktree's Git command or a workflow child whose launch outcome is unknown). The copy does not change after `UnprovenRetention.prove` removes the record, because the claim stays held.
- Three call sites use it, each falling back to its ordinary busy refusal:
  - `OrdinaryTurn.admit`: the Console's Send picks its copy with `activity_refusal_copy` (`Chat/console_ordinary_tree.py`).
  - `OrdinaryTurn.widen` (it excludes the turn's own claim): `widen_refusal` names the restart. The spawn admission in `agent_run_tools` routes this refusal to it too, under the tool the model called (`send_to_agent` for a resume).
  - `_InitializationSource.admit`: the `RESTART_REQUIRED` readiness.
- The refusal applies where an ordinary turn takes a claim: on platforms with native process groups (macOS and Linux).
- `CleanupUnprovenRefused` now has two meanings. From `reserve_tool`, the owner's own cleanup is unproven and its run ends (TASK-40716). From this predicate, another owner's claim is held and the run goes on. The run-ending handlers wrap only `reserve_tool`.
- Nothing in the running process retires the claim. `retain_unproven` marks the owner's drain unproven, and `WorktreeActivity._retire_if_settled` retires a claim only when every member's drain is proven. Even `UnprovenRetention.prove`, which no production code calls, lets the custody worker return but leaves the claim held.
- Pinned by:
  - `test_a_retained_cleanups_claim_refuses_later_console_sends_until_restart`, for an initialization job, a turn's retained tool operation, a sub-agent member's and an owner marked unproven, with an ordinary busy conflict as the control;
  - `test_a_retained_claim_that_a_send_does_not_need_keeps_the_busy_copy`, a held bounded claim the Send does not conflict with (both in `Tests/Workspaces/test_native_change_review_initialization.py`);
  - `test_a_claim_whose_owner_is_marked_unproven_holds_roots_until_restart` (same file), the initializer's `RESTART_REQUIRED` with no retained record;
  - `test_spawn_refused_by_a_retained_cleanups_claim_names_the_restart` (`Tests/Chat/test_console_ordinary_tree_widening.py`), with a live claim as the control, and `test_a_refused_child_launch_names_the_tool_the_model_called` (same file).

## Ask compaction hold amendment (TASK-33621.153, 2026-11-19)

This narrows "Elapsed time includes queueing and human approval waits" above for one pause.

**Owner decision, 2026-11-12, re-confirmed 2026-11-19 ([TASK-33621.153](../tasks/task-33621.153%20-%20Console-the-automatic-work-wall-clock-keeps-running-while-an-agent-wake-is-paused-at-the-Ask-trigger.md)).**
Pause the clock. Time a chain waits on a `compaction_ask` hold (TASK-33621.80: a wake refused before acceptance at the Ask compaction trigger) does not count against its automatic-work wall clock. The chain's time limit measures only the time the work runs, so a `/compact` after a long wait resumes normally.

**What still counts (unchanged).** Queueing, tool approval waits, `autowake_disabled` and every other pause code. A goal run's elapsed deadline also still counts its pauses, approvals and waits; a workflow already counts only active intervals.

**Implementation note (not an owner decision).**
- No schema change. Pausing a chain with `compaction_ask` stamps the hold's start in the chain's `last_observed_at`. It also folds the elapsed time so far (the larger of the wall and the in-process monotonic reading, as admission counts it) into `started_at`/`deadline_at`.
- The hold ends at the chain's next admission, or when `pause()` or a refused admission replaces it with another pause code. Settle's `usage_unknown`/`tokens_budget`, a chat start's `interrupted_work` and `recover` write the status directly and leave the hold's start in place, so the held time counts there; those states are final for the allowance. When the hold ends, `started_at` and `deadline_at` move forward by the held wall time, the monotonic anchor is re-based in the current process, and the stamp moves to that moment. So the end applies once, and two ends never exclude the same interval.
- Any admission in the allowance ends the hold, including other automatic work under the same allowance root, for example a wake in a chat the agent started. From then the clock runs again, and the rest of the held chat's wait counts, so a late `/compact` can still be refused as `wall_budget`. This keeps the other work's running time counted. Starting the hold again once that work settles is TASK-33621.157.
- The running time is the same before and after the hold whatever either clock did during it: a sleep, a forward step or an app restart (the stamp is durable). A backward step still refuses as `clock_reversed` and needs review.
- A refused first admission after the hold keeps the moved anchors with the stamp it already kept, so the hold is neither lost nor excluded twice. A clock that had not started stays unstarted, and a chain's first wake is unaffected.
