# ADR-141: Native Console goal runs

Status: Accepted
Date: 2026-09-08
Implementation status: Five slices approved; whole-branch review and final fixes pending. See the [qualification report](../../Docs/superpowers/reviews/2026-09-09-goal-runs-qualification.md) for evidence and limitations.
Tasks: TASK-32116 through TASK-32120 track the five-slice implementation.
Design: [gnhf review and native goal runs](../../Docs/superpowers/specs/2026-09-08-gnhf-inspired-goal-runs-design.md)
Plan: [First native goal-run milestone](../../Docs/superpowers/plans/2026-09-08-gnhf-inspired-goal-runs.md)
Review: [Preimplementation findings and corrections](../../Docs/superpowers/reviews/2026-09-08-goal-runs-preimplementation-review.md)
Amends: ADR-134/135 for a new goal-origin automatic attempt; preserves their fleet semantics.

## Context

The user requested a workflow inside Chatbook similar to gnhf. The inspected
reference repeatedly executes small agent contributions toward a durable objective.
Chatbook already owns native agent turns, tool permission gates, app-headless
Console execution, CLI/skill subprocess tools, configured MCP/ACP commands,
private run logs, shadow-Git review, and finite automatic-work accounting.
It lacks a durable outer goal lifecycle and evidence-based decision
about whether to continue.

The Workflows editor and sequential local engine are separately approved in
ADR-138. Their first local milestone is existing work. Adopting gnhf wholesale
would add a second orchestration/accounting owner and Git-centric effect recovery.
Existing CLI capability is available to the proposed native runner; there is no
requirement to build CLI support or postpone verification first.

## Decision

1. Add app-owned `GoalRunService` under `ConsoleRuntime`. It schedules bounded
   native iterations through the existing controller/bridge. It is independent
   of screen lifetime and does not replace the agent model/tool loop.
2. Persist goal/iteration/report state in AgentRunsDB. Extend the existing
   automatic-work ledger for typed `goal_iteration` attempts; do not manufacture
   survivor-run claims or create a parallel allowance store.
3. Add an explicit `GOAL_ITERATION` submission origin and a private typed
   authorization. All goal iterations, including the first, consume one causal
   chain's automatic allowance. Required FULL-synchronized acceptance precedes
   billable preparation and effect dispatch. Existing manual runs retain their
   policy, and ordinary fleet-wake claims retain their existing requirements.
4. Keep the model report advisory. Runtime-owned evidence and recorded completion
   checks determine progress; criteria needing judgment require a human result
   review. `RunOutcome.status == 'done'` means a finished turn, not a completed
   objective. Bind objective checks to a launch-approved verifier and input
   scope. Record typed process outcomes before rendering tool text;
   `ToolResult.ok` does not mean exit zero. Invalidate checks after artifact or
   verifier changes, including changes before human review. The immutable
   `human_review_required` launch flag defaults to true. Explicit false requires
   at least one selected verifier and makes those checks sufficient for objective
   completion; neither prose nor the model can infer or alter this choice.
5. Snapshot provider, resource bindings, criteria and finite policy at launch.
   Revalidate authority on each iteration and after waits. Tool allowlists narrow
   existing permissions; they do not grant them. Enforce the goal's scope on
   catalog tools, runtime callbacks and progressive loading at both schema and
   invocation boundaries. Automatically loaded project instruction bodies
   retain their ephemeral ownership under ADR-069.
6. Reuse existing budget reservations, exact-request accounting, manual capacity
   reserves and retained physical ownership. Start with one active goal and no
   goal-created subagents. Defaults are 3 iterations, 32 calls, 500000 budget
   tokens, 8192 maximum output tokens per call and 900 elapsed seconds. Each
   iteration also has at most 8 model turns, 64 steps and 240 seconds, narrowed
   by remaining allowance and applicable executor limits. Use distinct
   `goal_runs_enabled` and `max_goal_*` settings; keep `autowake_enabled` and
   `max_autowake_*` fleet-specific. Resolve policy by immutable origin at every
   context/dispatch check. Live settings may narrow, but never refill, a chain.
7. Preserve counters and pending evidence across retries/restart. Accepted work
   with uncertain outcome requires review; a new process does not automatically
   replay it. First-release restart requires explicit Resume even from a clean
   checkpoint. A pause does not extend the ADR-134 elapsed deadline. Share one
   startup audit and owner identity across fleet and goal coordinators; goal
   service construction must not trigger a second global recovery. Separate
   `awaiting_result_review` from `recovery_required`: quality approval cannot
   release physical ownership, clear uncertain charges or authorize replay.
8. Keep reports bounded and private. Use compact deterministic handoff memory;
   resolve evidence against exact run/source ownership. Missing history cannot
   be presented as successful verification. Referenced active evidence must
   survive source/artifact pruning through bounded private copies. Limits are
   128 KiB per evidence record, 4 MiB evidence per goal and a default 128 MiB
   aggregate goal-payload allowance, reserved before execution. Explicit
   removal of settled payloads retains accounting tombstones and protects
   active/uncertain records. This does not bound unrelated native logs.
9. Include existing CLI/script/MCP tools in the first native slice. Demonstrate
   a failed real validation command, an authorized file correction and a passing
   rerun, preserving outputs and diffs. Workflows may later invoke the same
   service through an advertised target capability; that does not establish
   portable/server support or change ADR-138's control-flow roadmap.
10. Reuse current execution services and grants. ADR-033's restriction on a
    particular native raw-shell interface is not a ban on CLI execution.
    Repository commit/worktree automation needs explicit checkpoint policy;
    it does not require adding basic CLI support. Existing shadow snapshots
    remain review evidence, not isolation or rollback of arbitrary effects.
    A script's scratch cwd is not filesystem/network confinement; preserve
    its existing trust, POSIX support and actual cleanup guarantees.
11. Persist a launch intent with its preallocated conversation ID and chain,
    then provision that exact conversation and workspace membership through
    their existing owners. Recover partial setup idempotently across stores;
    admit no iteration until ready. After execution, commit checkpoint,
    retained evidence, decision and attempt completion together in the same
    FULL transaction. Repeated identical results are idempotent; conflicting
    results fail without authorizing another iteration. Only conclusively
    settled work completes its attempt; saved partial evidence cannot settle
    an uncertain effect or release its conservative charge.
12. Build a fresh authorized request history for each iteration while retaining
    the full transcript for inspection. Appending a compact handoff to normal
    conversation history is insufficient. Do not restore a settled prior
    iteration's provider continuation. Keep continuation within an iteration
    coherent and account for the final request after all context preparation.

## Alternatives

| Alternative | Reason not selected |
| --- | --- |
| Launch gnhf as the first implementation | Introduces Node/agent-CLI distribution, separate authority and accounting, and Git-specific recovery for a broader application. |
| Add a generic loop/DAG engine | Duplicates the existing Workflows initiative and broadens its agreed staged scope. |
| Repeatedly submit ordinary user messages | Hides machine origin, can obtain fresh manual allowances, and interferes with draft/priority semantics. |
| Reuse wake attempts by inventing completed children | Violates the survivor ownership and result-claim invariants in ADR-135. |
| Treat agent-reported success as verification | Cannot distinguish claimed work from actual evidence or human quality judgment. |
| Reset failed increments with Git | Does not undo database or remote effects and can overwrite independent user work. |

## Consequences

The first deliverable requires no new external runtime dependency and can run
against existing native providers. The most significant work is extending
admission and recovery correctly, rather than writing the repetition loop.

Finite automatic defaults may pause a useful goal before it is complete. The
product must preserve progress and make that reason clear. A hard currency cap,
exactly-once external effects, machine-wide quotas and execution while the app
is closed are not capabilities established by this decision.

AgentRunsDB needs coordinated schema migration and retention changes. Goal bodies
are private payload; automatic ledger and diagnostic projections remain body-free.
The first implementation must reconcile the existing uncommitted agent-budget
changes and avoid taking ownership of the in-progress Workflows implementation.

The number was checked against 468 local branch/remote refs and 19 registered
worktrees; the highest observed existing ADR was 140. This is a local snapshot,
not a reservation against concurrent or unfetched work. Recheck before acceptance
or integration and add the accepted decision to the ADR index at that point.

## Related decisions

- [Local tool permissions](032-local-agent-tool-permission-boundary.md)
- [Local process boundary](033-local-agent-process-execution-boundary.md)
- [Existing skill subprocess execution](../../tldw_chatbook/Skills_Interop/skill_script_runner.py)
- [Existing agent script dispatch](../../tldw_chatbook/Chat/console_agent_bridge.py)
- [Existing MCP executable transport](../../tldw_chatbook/MCP/client.py)
- [Provider continuation](063-hosted-provider-wire-and-durable-tool-continuation.md)
- [Human approval waits](067-indefinite-human-approval-waits.md)
- [Project instruction ownership](069-console-project-instruction-local-state-and-preflight.md)
- [Budget accounting](131-durable-agent-budget-accounting.md)
- [Automatic admission](134-fleet-admission-and-automatic-work-budgets.md)
- [Automatic recovery](135-fleet-completion-delivery-and-crash-recovery.md)
- [Portable Workflows](138-portable-workflow-definitions-and-local-execution.md)
