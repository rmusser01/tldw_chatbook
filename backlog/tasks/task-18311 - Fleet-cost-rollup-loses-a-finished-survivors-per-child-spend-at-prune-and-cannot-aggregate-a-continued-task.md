---
id: TASK-18311
title: >-
  Fleet cost rollup loses a finished survivor's per-child spend at prune and
  cannot aggregate a continued task across its two runs
status: To Do
assignee: []
created_date: '2026-08-18 15:40'
labels:
  - agents
  - console
priority: medium
dependencies: []
---

## Description (the why)

Filed by PR 3b Task 6's cost-ticker audit (spec
`2026-08-08-supervisor-agent-fleet-design.md` §8's 3b row, executed at dev
`cf5db6f50`), per the plan's explicit "FILE the follow-up, do not patch it
here". Two honest gaps in the fleet-facing spend surfaces, both verified in
code, neither a regression:

1. **A finished survivor's per-child spend leaves the fleet rollup between
   turns.** `ConsoleAgentController._console_agent_fleet_token_total`
   (`UI/Console_Modules/agent.py`) sums `FleetHandle.total_tokens` over
   `bridge.fleet_snapshot(conversation_id)`; `total_tokens` is recorded only
   at `FleetCoordinator.finish`, and the bridge's turn-start
   `coordinator.prune_terminal()` (`Chat/console_agent_bridge.py`, the
   turn-start prune) drops every terminal handle — so a finished child's
   figure is visible on the panel row and in the `fleet_tokens` aggregate
   only from its finish until the next turn starts, then vanishes. The
   CHIP-level money story is separately covered (`cced002ab`'s
   `unattributed_fleet_tokens` "Sub-agents: N tok (not priced)" line, and
   3a-2's `FleetDrained` usage re-attach fold, task-15660) — the gap is the
   per-child figure and the fleet aggregate, not the billed total.
2. **A continued task's aggregate spend spans two runs no surface can
   join.** A resumed child (PR 3b Task 4) is a NEW run whose handle records
   only the NEW run's spend; the OLD run's figure died with its pruned
   handle. `agent_runs` persists NO token column (verified: no token field
   in `DB/AgentRuns_DB.py`), so the DB can join the lineage via
   `resumed_from_run_id` but cannot aggregate the spend; `fleet_snapshot`
   can do neither. Only the primary's run-log manifest carries a
   `total_tokens` figure, and that is the primary's own.

The audit's positive half is pinned:
`Tests/Agents/test_fleet_continuation.py::
test_a_resumed_childs_spend_reaches_the_fleet_rollup_at_finish` proves a
resumed child's `total_tokens` reaches the same rollup as any child, and the
characterization half of that test pins gap (1) with a comment citing this
task.

## Acceptance Criteria (the what)

- [ ] A continued task's aggregate spend (original + resumed runs) is derivable from at least one durable surface (e.g. a per-run token column joined over `resumed_from_run_id`), or an owner decision records that per-run spend is deliberately ephemeral
- [ ] A finished survivor's per-child spend either survives the next turn's prune on some fleet-facing surface, or the between-turns-only visibility is documented in the User Guide and pinned by a characterization test
- [ ] No double counting against the chip's existing `unattributed_fleet_tokens` / drain re-attach story (task-15660)
