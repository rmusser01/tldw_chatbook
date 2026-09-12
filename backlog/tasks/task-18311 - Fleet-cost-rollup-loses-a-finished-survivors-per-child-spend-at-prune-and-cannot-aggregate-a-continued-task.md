---
id: TASK-18311
title: >-
  Fleet cost rollup loses a finished survivor's per-child spend at prune and
  cannot aggregate a continued task across its two runs
status: Done
assignee:
  - '@codex'
created_date: '2026-08-18 15:40'
updated_date: '2026-09-08 04:24'
labels:
  - agents
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
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
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A continued task's aggregate spend (original + resumed runs) is derivable from at least one durable surface (e.g. a per-run token column joined over `resumed_from_run_id`), or an owner decision records that per-run spend is deliberately ephemeral
- [x] #2 A finished survivor's per-child spend either survives the next turn's prune on some fleet-facing surface, or the between-turns-only visibility is documented in the User Guide and pinned by a characterization test
- [x] #3 No double counting against the chip's existing `unattributed_fleet_tokens` / drain re-attach story (task-15660)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-09-07-durable-agent-budget-accounting.md using failing regressions before production edits. Preserve budget-counter semantics and keep the cost-chip/provider-usage path separate.

ADR required: yes
ADR path: backlog/decisions/131-durable-agent-budget-accounting.md
Reason: Schema, nullable accounting, continuation aggregation, and fleet presentation contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed durable budget accounting under ADR-131 (backlog/decisions/131-durable-agent-budget-accounting.md). AgentRunsDB schema v16 stores nullable per-run budget_tokens, with guarded migration and a SQL reference artifact. First-known accounting can arrive after cancellation without replacing terminal status/result or counting twice. Existing legacy rows and synthesized exception outcomes remain unknown.

AgentService persists the actual weighted/estimated RunOutcome counter. Historical fleet rows display known budget usage (including zero), while continued-run detail shows scoped ancestry totals and explicit partial accounting; sibling forks and foreign runs are excluded. The existing live-fleet cost feed and ProviderUsage billing/reattachment computation remain unchanged. The panel still presents its existing latest-run scope; historical elapsed/result detail and navigation belong to TASK-15200/15201.

Validation: 656 targeted tests passed in disjoint groups (282 DB/service/runtime, 288 bridge/usage/cost/lifecycle, 86 UI). Two strengthened, overlapping compositor tests passed after verifying actual handle prune and a new primary record preserve selected-run accounting while the live feed becomes zero. Fifteen DB tests passed again after fixture cleanup. Initial regressions reproduced the missing persistence/UI behavior. New files pass Ruff lint/format; changed-range formatting and scoped whitespace pass; six existing source/test files add no lint findings versus the pass-start working tree. Persistent diagnostic inventory passes unchanged. No full suite or live provider was run. Existing requests dependency warning remains.

Updated Docs/User_Guide/console/agent-runs-and-tools.md, the review ledger, ADR index, and Docs/superpowers/plans/2026-09-07-durable-agent-budget-accounting.md. Self-review covered late writes, unknown-vs-zero, incomplete/foreign/cyclic ancestry, SQLite overflow, units, and billing isolation. Changes remain uncommitted; unrelated working-tree changes preserved.
<!-- SECTION:NOTES:END -->
