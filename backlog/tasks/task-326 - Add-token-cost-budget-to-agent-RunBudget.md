---
id: TASK-326
title: Add a token/cost budget to the agent RunBudget
status: Done
assignee: []
created_date: '2026-07-20 18:45'
labels: [agents, cost]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The agent runtime bounds steps, model-turns, wall-clock, and sub-agent count (`agent_models.py:88-111`) but has no token or dollar budget (grep for `token_budget`/`cost_budget`/`max_tokens` in `Agents/` returns nothing). A run on a large-context model with big transcripts can burn substantial tokens within the turn cap with no spend ceiling and no accounting to alert or stop on cost. Providers already return usage, so it can be tracked.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `RunBudget` gains a `max_total_tokens` (and/or cost) budget
- [x] #2 The loop accumulates prompt+completion tokens per run and stops cleanly (like the existing caps) when the budget is exceeded
- [x] #3 Reaching the budget produces a clear terminal status/step, not a silent truncation
- [x] #4 Tests cover budget exhaustion
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a **token-spend ceiling** to the agent RunBudget — the documented complement to task-322 (322 bounds conversation history INTO the agent loop at dispatch; 326 bounds the loop ITSELF, the tool-call/result messages accumulating across iterations). Decision (user-approved): tokens-only (`max_total_tokens`); a dollar-cost budget is deferred (needs a per-model price table that doesn't exist yet).

Approach (3 units, all mirroring the existing step/model-turn/wall-clock cap machinery so behavior is byte-identical until a consumer opts in):
- **`agent_models.py`**: `RunBudget.max_total_tokens: int = 0` (**0 = unlimited sentinel**); `ModelTurn.tokens` (prompt+completion for one provider call); `RunOutcome.total_tokens` (measured run spend, for the "accounting" the description asked for); `clamp_child_budget` passes the field through (each run, parent or sub-agent, has its own budget).
- **`agent_runtime.py` `run_agent_loop`**: accumulate `total_tokens += turn.tokens` per model turn; a top-of-loop check (after wall-clock) `if budget.max_total_tokens and total_tokens >= budget.max_total_tokens:` → `add(STEP_ERROR, "token budget exhausted")` + `RUN_STUCK` (AC#3, a clear terminal step, not silent truncation). An `_outcome()` helper closure routes all 8 terminal returns so spend is always reported. Checking at the TOP means a run that produces its final answer on the crossing turn still completes `RUN_DONE` (stop-the-loop, not fail-the-answer). The `and` guard is deliberately fail-closed: a negative (invalid) budget trips immediately rather than reading as "unlimited".
- **`agent_service.py` `_make_call_model`**: **hybrid** token source — real `resp["usage"]` (via `_usage_total_tokens`) when the provider reports it, else the task-321 estimate `count_tokens_messages(payload) + estimate_tokens(text)`. Populated on both `ModelTurn` return paths (native + non-native).

`max_total_tokens` is cumulative SPEND (Σ prompt+completion across turns, since the growing prompt is re-sent each call), NOT a context-window size — a consumer must size it as a spend ceiling (hundreds of thousands).

Testing: full agent suite 275 passed. Loop tests (exhaustion→STUCK, 0-sentinel never trips, DONE-on-crossing-turn boundary, total_tokens accounting); `_usage_total_tokens` (total / prompt+completion / None-on-malformed, never crashes); call_model hybrid on both native and non-native paths. Executed via SDD (implementer+reviewer per task, opus whole-branch review Ready-to-merge, 0 Crit/0 Imp).

Deferred follow-up minors (non-blocking, from reviews): `_usage_total_tokens` doesn't reject `bool`/negative token counts (no real provider emits them); error-path `RunOutcome` (`RUN_ERROR`) reports `total_tokens=0` (partial pre-crash spend lost); `total_tokens` is return-only, not persisted to AgentRuns_DB (surfacing in a consumer UI/DB was out of scope).

Files: `tldw_chatbook/Agents/agent_models.py`, `tldw_chatbook/Agents/agent_runtime.py`, `tldw_chatbook/Agents/agent_service.py`, `Tests/Agents/test_agent_models.py`, `Tests/Agents/test_agent_runtime.py`, `Tests/Agents/test_agent_service.py`.
<!-- SECTION:NOTES:END -->
