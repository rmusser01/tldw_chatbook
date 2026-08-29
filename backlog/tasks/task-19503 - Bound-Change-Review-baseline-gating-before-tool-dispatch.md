---
id: TASK-19503
title: Bound Change Review baseline gating before tool dispatch
status: Done
assignee:
  - '@codex'
created_date: '2026-08-21'
updated_date: '2026-08-21 18:54'
labels:
  - console
  - tools
  - concurrency
dependencies:
  - TASK-19502
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent tracked workspace mutations from racing ahead of their Change Review baseline without turning review into authorization or allowing a cold file scan to wedge an agent turn.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Project-instruction preparation and the existing permission review run before any baseline wait
- [x] #2 Explicit non-proceed review verdicts skip baseline waiting and dispatch while existing refusal copy, stamps, audit, and invocation ownership remain unchanged
- [x] #3 Every remaining potentially dispatchable provider, skill, script, spawn, message, and unknown call waits across all tracked roots with a three-second bound
- [x] #4 Only the approved pure runtime discovery and status tools bypass baseline waiting
- [x] #5 A raised review hook cannot bypass the bounded all-roots wait before the runtime's existing hook-failure policy continues
- [x] #6 Timed-out roots are irrevocably untracked for the turn and late baselines cannot restore or publish misleading diffs
- [x] #7 A survivor plus successor-baseline timeout invalidates both windows, enters a degraded epoch, and resynchronizes only after quiescence without false attribution
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a generic runtime pre-dispatch seam after project-instruction preparation and permission review, passing only calls whose effective verdict is `proceed`.
2. Move Change Review's baseline wait from the permission-review wrapper into that seam; bypass only the fixed pure runtime-tool table and bound every other batch to three seconds.
3. Make timeout reporting immutable and alias-safe, reject late baseline success, and persist the existing per-root tracking-error records without changing authorization/refusal ownership.
4. Teach the root coordinator to invalidate an open predecessor survivor window when a queued successor times out, hold a degraded root epoch across known mutation/survivor activity, and resume ordinary tracking only after quiescence.
5. Add deterministic ordering, raised-review, refusal, bypass, timeout, late-result, survivor-degradation, and recovery tests; run focused runtime/bridge/coordinator suites plus Ruff and diff checks.

ADR required: no
ADR path: `backlog/decisions/084-change-review-consent-and-asynchronous-finalization.md`
Reason: ADR-084 already defines the conservative bounded pre-dispatch gate, timeout invalidation, survivor degradation, and recovery boundary implemented here.

Detailed plan: `Docs/superpowers/plans/2026-08-21-change-review-bounded-baseline-gate.md`
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a generic runtime hook after project-instruction preparation and effective permission review, then moved Change Review's baseline gate to that hook. Only actually wired members of the fixed eight-tool pure-runtime table bypass the gate; all other proceed calls share one three-second wait per turn, and raised review hooks still reach the gate before the existing failure policy dispatches them.

Timeouts now invalidate affected current, predecessor, nested-root, and survivor windows; late baseline results cannot restore attribution. Canonical roots remain degraded until every timed-out mutation reservation and known survivor settles, after which fresh baselines resume. Dynamic nested-root enrollment preserves sequence for pre-baseline work while refusing to overtake B-in-progress, post-B, finalizing, or survivor windows, preventing both false attribution and multi-root lane cycles. User-visible warnings disclose workspace aliases only.

Verification: 188 coordinator/runtime/service tests and 76 Console/context/consent tests passed; the three nested-root ordering regressions passed ten consecutive runs. Ruff, `py_compile`, and `git diff --check` passed for the changed files. Independent final review reported no Important findings.

ADR required: no. ADR-084 already defines the bounded gate, timeout invalidation, degraded recovery, and asynchronous finalization policy implemented here.
<!-- SECTION:NOTES:END -->
