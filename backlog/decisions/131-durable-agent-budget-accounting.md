# ADR-131: Durable agent budget accounting

Status: Accepted
Date: 2026-09-07
Related task: TASK-18311
Related review: [Agent orchestration review](../docs/agent-orchestration-review-2026-09-07.md)

## Decision

AgentRunsDB schema v13 adds nullable `agent_runs.budget_tokens`. This preserves
the completed run's existing `RunOutcome.total_tokens` counter. The UI calls it
**budget tokens**: the runtime may estimate missing provider usage and weight
cache buckets. It is neither a raw provider-token count nor a currency amount.
Old rows remain NULL (unknown); no transcript or step-text reconstruction invents
historical totals. A measured zero is stored as zero.

`set_status(..., budget_tokens=...)` stores the first known nonnegative SQLite
integer alongside the status transaction. Terminal status/result remain
first-writer-wins. A later completed child may fill an unknown budget counter
after cancellation or supersession without replacing that status/result; a
repeated outcome cannot add or overwrite spend. The return value continues to
report whether status was updated. Synthesized outcomes after an exception
escaping the runtime do not turn unknown spend into zero.

The DB exposes a conversation-scoped continuation-ancestry query. It includes
the requested sub-agent and each `resumed_from_run_id` ancestor once, excluding
sibling forks and primary runs. A recursive UNION prevents cycles from looping.
The result carries the sum of recorded budget tokens, run and recorded-run
counts, and completeness. Completeness requires a root, known counters, and
terminal runs throughout the chain. Missing/foreign parents, cycles, and legacy
NULL counters produce an explicitly partial total. Integer accumulation occurs
in Python so a valid chain cannot overflow SQLite's SUM accumulator.

The bridge passes durable per-run counters into historical fleet rows and
supplies ancestry accounting for a selected continued run. The drill-in shows
the per-run figure and full or partial chain figure. The existing live-fleet
cost-chip feed and provider-usage reattachment remain separate: persisted
history is never fed back into that billing path or added to its totals.

## Alternatives and consequences

- Keeping only ephemeral handle counters loses completed history at prune and
  prevents continuation accounting. Documentation alone does not repair it.
- Calling this column `total_tokens` would silently present weighted or
  estimated budget units as raw provider usage. A separately named field keeps
  the existing runtime semantics explicit.
- Rebuilding billed money from this counter would mix models, rates, cache
  weights, and estimates. ProviderUsage remains the billing source.
- Summing all descendants would count sibling forks as if they were the selected
  continuation. An ancestry total describes the selected work path; each fork
  can include the shared original when viewed independently, and these path
  totals must not be summed as a conversation-wide bill.

The migration uses this DB's existing guarded ALTER mechanism, increments its
constant and version table together, and includes a SQL migration reference in
`DB/migrations/`. Runtime budget limits and aggregate admission policy do not
change. A crash before a completed outcome is persisted leaves unknown usage;
incremental billing checkpoints and global/wake-chain budgets remain separate.
