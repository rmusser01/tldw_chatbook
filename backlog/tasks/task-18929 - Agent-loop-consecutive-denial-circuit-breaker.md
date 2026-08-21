---
id: TASK-18929
title: 'Agent loop: consecutive-denial circuit breaker'
status: To Do
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-08-19 09:55'
labels:
  - agents
  - tools
  - approvals
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's consecutive-denial circuit breaker (2026-08-19 hermes-release review). When an agent's tool calls are denied several times in a row, it often re-asks forever — burning turns and tokens. Add a configurable circuit breaker: after N consecutive denials within a run (default small, e.g. 3), the loop stops with an honest terminal message ("stopped after N consecutive denied tool calls — review the denial reasons or rephrase") instead of continuing to re-ask. The counter resets on any successful or approved call; only unbroken denial streaks trip it. Applies per run, including per-child in the fleet (a child's denials trip that child's breaker, not a sibling's).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Config knob (e.g. `[agents] denial_circuit_breaker_limit`) with a documented default; 0 or absent disables the breaker cleanly
- [ ] #2 The streak counter resets on any successful/approved tool call; only consecutive denials trip the breaker — pinned by tests
- [ ] #3 Tripping produces a clear terminal state: transcript System row + run-log record naming the count; the user can immediately retry or continue (no silent hang, no lost partial reply)
- [ ] #4 Fleet isolation: the breaker is per run/child; one child tripping does not stop a sibling or the supervisor
- [ ] #5 Tests cover trip, reset, disabled mode, per-child isolation, and the terminal messaging
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: bounded loop guard within the existing agent runtime; no schema/boundary change (config knob follows existing `[agents]` conventions).

1. Streak counter in the agent loop's denial path (shared with the child runtime)
2. Terminal-state handling + transcript/run-log messaging
3. Config knob + docs (agent-runs-and-tools.md "Related settings")
4. Tests per the AC matrix
<!-- SECTION:PLAN:END -->
