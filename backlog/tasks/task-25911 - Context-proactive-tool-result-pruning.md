---
id: TASK-25911
title: 'Context: proactive tool-result pruning'
status: To Do
assignee: []
created_date: '2026-08-31 15:10'
updated_date: '2026-08-31 15:11'
labels:
  - console
  - context
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Chatbook reclaims context only by dropping whole turn groups. Verified on origin/dev: Chat/console_history_budget.py:266 bound_messages_to_window drops turn-group-aware whole units, and a named grep for tool_result pruning and strip tool output across Chat/ and Agents/ returns zero - a large old tool result is either fully present or the whole turn is gone. Hermes prunes large stale tool results deterministically with no LLM call and a minimum-reclaim gate so prompt-cache breaks stay episodic. Cheapest real token win available, because it needs no model call and no new storage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Large tool results older than a configurable recency threshold are shrunk in place, keeping a bounded head plus a statement of what was removed
- [ ] #2 Pruning requires no LLM call
- [ ] #3 A minimum-reclaim threshold prevents pruning that would break the prompt cache for negligible gain
- [ ] #4 The most recent N turns are never pruned, so the model always has its immediate working context intact
- [ ] #5 Pruning is visible in the context accounting rather than silently changing the numbers
- [ ] #6 Disabled by config reproduces today's behavior exactly
<!-- AC:END -->
