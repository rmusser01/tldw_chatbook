# Durable agent budget accounting implementation plan

**Goal:** Complete TASK-18311 by preserving per-run budget usage and exposing
continuation ancestry totals without double-counting billed usage.

**Architecture:** Add nullable accounting to AgentRunsDB, persist it when an
outcome is available, and project it into existing historical rows and drill-in
copy. Keep the live-fleet billing feed and ProviderUsage reattachment unchanged.

**Tech stack:** Python, SQLite, Textual, pytest.

**Spec:** backlog/decisions/131-durable-agent-budget-accounting.md

ADR required: yes
ADR path: backlog/decisions/131-durable-agent-budget-accounting.md
Reason: Durable schema, metric semantics, continuation aggregation, and UI/DB
interfaces require an explicit owner decision.

## Database and persistence

- [x] Add red DB tests for legacy migration/reopen, unknown versus zero,
  cancellation followed by one late budget write, repeated outcome idempotence,
  invalid counters, and scoped continuation chains with branches/cycles/missing
  parents. Expect ancestry 100 + 40 + 10 = 150 while a sibling's 900 is excluded.
- [x] Bump AgentRunsDB to v13, add nullable `budget_tokens`, guarded ALTER and
  version-table row, plus `DB/migrations/agent_runs_v12_to_v13_budget_tokens.sql`.
- [x] Extend `set_status` with optional keyword-only `budget_tokens: int | None`.
  Validate non-bool integers in 0..2**63-1; write first known value under the same
  transaction as the existing guarded status update.
- [x] Add `continuation_budget(conversation_id, run_id) -> dict | None`, selecting
  only ancestry IDs/statuses/counters through a recursive UNION and accumulating
  in Python. Return budget_tokens/run_count/recorded_run_count/complete.
- [x] Make `AgentService._persist` forward the actual outcome counter; mark
  synthesized exception outcomes unknown. Add real-service tests for primary,
  continued child, cache-weighted accounting, and no fabricated zero on failure.

## Fleet presentation

- [x] Add red rendered tests for historical budget tokens after prune/reopen,
  known zero versus unavailable usage, and full/partial continuation copy.
- [x] Add optional `budget_tokens` to SubAgentSummary and populate historical
  summaries from DB rows. Enrich selected continued-run records with the scoped
  ancestry result. Use explicit budget-token labels in live/history/drill-in.
- [x] Keep `_console_agent_fleet_token_total`'s computation unchanged; verify
  history is not reintroduced into cost-chip totals after prune.
- [x] Update the existing continuation characterization to assert persisted
  per-run/chain counts survive prune while its live-only sum still becomes zero.

## Verification and completion

- [x] Run the new tests first, observe intended failures, then run DB/service/
  continuation/bridge/fleet UI and usage-reattachment/cost tests as targeted groups.
- [x] Self-review nullable/late-write semantics, cross-conversation isolation,
  bounded data projection, and numeric overflow. Check lint/format for new files,
  changed-range formatting, no added legacy lint findings, and scoped whitespace.
- [x] Update the user guide, review ledger, task notes and checked criteria;
  mark Done through Backlog CLI only after verification. Preserve unrelated
  working-tree changes; no full suite, live provider, or commit is requested.
