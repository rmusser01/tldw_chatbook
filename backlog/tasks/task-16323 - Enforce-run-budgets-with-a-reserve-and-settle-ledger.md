---
id: TASK-16323
title: Enforce run budgets with a reserve-and-settle ledger
status: Done
assignee:
  - '@robert'
created_date: '2026-08-15 05:15'
updated_date: '2026-08-15 06:27'
labels:
  - research
  - budget
dependencies:
  - TASK-16322
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The runs schema carries limits_json but nothing enforces it and the deep-search pipeline bounds only time and bytes: phase 1 sub-query generation plus phase 2 map-reduce can make many LLM calls with unbounded spend. Port tldw_server's pattern: reserve before each unit of work, settle after, hard-stop when exhausted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 limits_json keys (max_searches and max_fetched_docs and max_runtime_seconds at minimum) are enforced before each unit of work with structured research_limit_exceeded errors
- [x] #2 Token reserve/settle infrastructure exists in the ledger and persists per run (AMENDED during implementation: `chat_api_call` and the summarization path return text only with no usage report, so honest per-LLM-call token enforcement is impossible until the pipeline reports usage — the ledger supports it, the engine does not fabricate counts)
- [x] #3 An exhausted budget stops the run cleanly at a phase boundary with a terminal event while preserving partial artifacts
- [x] #4 Ledger state is inspectable per run
- [x] #5 Tests pin the enforcement and clean-stop behavior
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Check run record limits shape (normalize limits_json) and whether the pipeline exposes token usage at all
2. TDD Research_Interop/research_budget.py: BudgetLedger with reserve/settle for searches and fetched docs and tokens, runtime deadline checks, non-negative invariants, structured ResearchLimitExceeded(key) errors, and a snapshot for persistence
3. TDD engine enforcement: parse limits_json into the ledger, check before each phase unit, pass the runtime budget into phase 1 via phase1_time_budget_s, stop cleanly through fail_run with a research_limit_exceeded message at a phase boundary while preserving partial artifacts, persist budget_ledger.json after settlements
4. Tests plus lint plus task close
ADR required: no - enforcement lands inside the ADR-068 engine contract which explicitly anticipated task-16323; link ADR-068
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- **Ledger** (`Research_Interop/research_budget.py`): `BudgetLedger` ports mole's reserve/settle discipline + the server's `limits.py` semantics. Missing/negative/NaN limits mean unlimited; an explicit 0 is a valid zero budget. Searches use reserve (net of outstanding reservations) then settle (overshoot vs reservation recorded, never an error); docs use `allot_docs` (caps a batch at remaining; raises on exhausted) then settle; runtime via `check_runtime()` between phases; `max_tokens` reservations capped and settled at the API level. `ResearchLimitExceeded` carries `limit_key` and stringifies as `research_limit_exceeded:<key>: ...` matching the server error contract. `snapshot()` is the persistable/inspectable form.
- **Engine enforcement** (`local_research_engine.py`): the run's `limits` build the ledger; runtime is checked at every phase entry; BEFORE phase 1 spends, the search fan-out cap is clamped into `search_default_max_queries` and the remaining runtime handed to the pipeline as `phase1_time_budget_s` (both knobs `generate_and_search` already honors); after collection, searches settle at the actual query count (1 + sub-queries) and the fetched-doc batch is truncated to the remaining doc budget before synthesis. Collection artifacts are saved BEFORE enforcement so a budget stop preserves the evidence of what was collected; a cap hit upserts `collection_summary.json` with a truncation note. Exhaustion resolves through `fail_run` with the structured message (terminal event) and the ledger persisted as `budget_ledger.json` — also persisted after every settle and on any failure path.
- **Token deviation (documented, AC amended)**: the pipeline's LLM calls (`chat_api_call`, `Summarization analyze`) expose no usage counts, so per-call token enforcement would require fabricating numbers. The ledger's token API is complete and wired for persistence; real enforcement waits on usage plumbing in the LLM call path (candidate follow-up task).
- Verified TDD: 9 ledger tests + 4 engine enforcement tests written first and watched failing; full `Tests/Research/` = 70 passed; ruff clean. Files: new `research_budget.py`, `test_research_budget.py`; modified `local_research_engine.py`, `test_local_research_engine.py`.
<!-- SECTION:NOTES:END -->
