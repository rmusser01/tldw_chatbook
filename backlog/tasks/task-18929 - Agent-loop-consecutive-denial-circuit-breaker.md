---
id: TASK-18929
title: 'Agent loop: consecutive-denial circuit breaker'
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-19 09:55'
updated_date: '2026-09-12 16:28'
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
- [ ] #1 The streak counter resets on any successful/approved tool call; only consecutive denials trip the breaker — pinned by tests
- [ ] #2 Tripping produces a clear terminal state: transcript System row + run-log record naming the count; the user can immediately retry or continue (no silent hang, no lost partial reply)
- [ ] #3 Fleet isolation: the breaker is per run/child; one child tripping does not stop a sibling or the supervisor
- [ ] #4 Tests cover trip, reset, disabled mode, per-child isolation, and the terminal messaging
- [ ] #5 The agents denial_circuit_breaker_limit setting defaults to 3; explicit 0 disables it and invalid values use the documented conservative default.
- [ ] #6 Only authoritative explicit user denial or configured permission Off increments the streak; unanswered, timeout, cancellation, stale authority, legacy refusal text and synthetic restored-pending results do not.
- [ ] #7 Evaluate the trailing streak after a fully settled tool batch, preserve every tool reply, and stop before another model call; an approved tail resets the streak and the terminal message reports the actual observed count.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/154-agent-denial-streak-boundary.md. Reason: compatible structured provider/review/runtime provenance and coherent stop boundary. Execute Docs/superpowers/plans/2026-09-12-agent-denial-breaker.md: compatible typed review facts; authoritative producers; run-local budget/counter; Console and real persistence/fleet evidence; docs and independent review. Accepted ADR-078 remains unchanged.
<!-- SECTION:PLAN:END -->
