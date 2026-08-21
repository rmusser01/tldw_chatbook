---
id: TASK-19503
title: Bound Change Review baseline gating before tool dispatch
status: To Do
assignee: []
created_date: '2026-08-21'
labels:
  - console
  - tools
  - concurrency
dependencies:
  - TASK-19502
priority: critical
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent tracked workspace mutations from racing ahead of their Change Review baseline without turning review into authorization or allowing a cold file scan to wedge an agent turn.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Project-instruction preparation and the existing permission review run before any baseline wait
- [ ] #2 Explicit non-proceed review verdicts skip baseline waiting and dispatch while existing refusal copy, stamps, audit, and invocation ownership remain unchanged
- [ ] #3 Every remaining potentially dispatchable provider, skill, script, spawn, message, and unknown call waits across all tracked roots with a three-second bound
- [ ] #4 Only the approved pure runtime discovery and status tools bypass baseline waiting
- [ ] #5 A raised review hook cannot bypass the bounded all-roots wait before the runtime's existing hook-failure policy continues
- [ ] #6 Timed-out roots are irrevocably untracked for the turn and late baselines cannot restore or publish misleading diffs
- [ ] #7 A survivor plus successor-baseline timeout invalidates both windows, enters a degraded epoch, and resynchronizes only after quiescence without false attribution
<!-- AC:END -->
