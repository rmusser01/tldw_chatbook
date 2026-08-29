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

One app-owned consent service serializes local turn admission, toggle commit
and readiness publication, and initializer completion per workspace. A turn
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
boundary before a successor window begins. Filesystem workers receive only
immutable inputs and return pure values; they never receive a database, Console
store, widget, or UI callback.

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

### Conservative, bounded pre-dispatch observation gate

Change Review remains observation, not authorization. Existing project-context
preparation, permission review, invocation-time permission resolution,
refusals, stamps, audit, and kill-switch ownership remain unchanged.

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
- [ADR-032: Local agent tool permission boundary](032-local-agent-tool-permission-boundary.md)
- [ADR-046: Visible bounded Console prompt queue](046-visible-bounded-console-prompt-queue.md)
- [ADR-069: Console project-instruction local state and preflight](069-console-project-instruction-local-state-and-preflight.md)
