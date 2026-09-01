---
id: TASK-25911
title: 'Context: proactive tool-result pruning'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:10'
updated_date: '2026-09-01 16:39'
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
- [x] #1 Large tool results older than a configurable recency threshold are shrunk in place, keeping a bounded head plus a statement of what was removed
- [x] #2 Pruning requires no LLM call
- [x] #3 A minimum-reclaim threshold prevents pruning that would break the prompt cache for negligible gain
- [x] #4 The most recent N turns are never pruned, so the model always has its immediate working context intact
- [x] #5 Pruning is visible in the context accounting rather than silently changing the numbers
- [x] #6 Disabled by config reproduces today's behavior exactly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: 6 pure-pruner tests (shrink+note, recency fence, fence protocol, min-reclaim identity, non-string skip, idempotency)\n2. prune_stale_tool_results + ToolResultPruneSettings/Stats in console_history_budget.py (pure, LLM-free, identity on no-op)\n3. AgentService seam: _prune_send_payload before bound_history_for_send at both payload sites, protocol-aware round boundary\n4. Config: [agents] prune_* keys, OFF by default; ctor override for tests; sample documented
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Pure prune_stale_tool_results in console_history_budget.py: groups the payload with the caller's turn/round boundary, protects the newest keep_recent_turns groups (AC#4), shrinks tool-result rows (native role:tool + fence 'Tool result for ' user rows; string contents only) larger than max(min_result_chars, head_chars) to a head + '[tool output pruned: kept first X of Y chars; Z chars removed…]' note (AC#1, and the marker doubles as the idempotency sentinel), and returns the INPUT OBJECT unchanged when total reclaim < min_reclaim_chars (AC#3 — prompt-cache break only when the win is real). No LLM anywhere (AC#2). AgentService applies it via _prune_send_payload before bound_history_for_send at both payload-build sites, with run_log_eviction's protocol-aware _make_round_boundary so native pairs never straddle the recency fence; stats land in a log line beside the in-row notes (AC#5). Config: [agents] prune_stale_tool_results (default OFF -> byte-identical today, AC#6, pinned by a wire-payload test) + 4 threshold keys, coerced fail-closed; explicit ctor settings override config (RunLogWriter pattern). 8 new tests (6 pure + 2 seam); seam mutation-verified; Tests/Agents exact 7-name baseline. Trade-off: default OFF (conservative; hermes defaults on) — flipping is one config line.
<!-- SECTION:NOTES:END -->
