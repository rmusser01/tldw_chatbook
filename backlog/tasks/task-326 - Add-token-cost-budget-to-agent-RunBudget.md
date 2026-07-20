---
id: TASK-326
title: Add a token/cost budget to the agent RunBudget
status: To Do
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
- [ ] #1 `RunBudget` gains a `max_total_tokens` (and/or cost) budget
- [ ] #2 The loop accumulates prompt+completion tokens per run and stops cleanly (like the existing caps) when the budget is exceeded
- [ ] #3 Reaching the budget produces a clear terminal status/step, not a silent truncation
- [ ] #4 Tests cover budget exhaustion
<!-- AC:END -->
