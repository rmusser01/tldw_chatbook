---
id: TASK-19505
title: Measure and reduce remaining Console first-interactive mount cost
status: Done
assignee:
  - '@codex'
created_date: '2026-08-21'
updated_date: '2026-08-21 19:46'
labels:
  - console
  - performance
  - diagnostics
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measure the remaining Textual-dominated Console mount floor by top-level subtree and keep a narrowly deferred secondary subtree only if it materially improves first interaction without lifecycle or input regressions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At least thirty isolated warm navigations record first-interactive paint, full-ready time, subtree widget/mount cost, focus restore, teardown, key-to-echo, and Enter-to-worker latency with median and p95
- [x] #2 Any production deferral reduces median first-interactive time by at least 15 percent while full-ready median regresses no more than 5 percent and input p95 regresses no more than 10 percent
- [x] #3 The retained change demonstrably reduces pre-interaction widget or mount work and every eager query, hook, focus, restore, and view binding tolerates the deferred subtree
- [x] #4 Fresh-screen, rapid-switch, focus, restore, unmount, and interactive Console soak gates pass
- [x] #5 If no candidate clears the thresholds, the task closes with reproducible measurements and no speculative production refactor
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Build a reproducible isolated-profile Pilot probe that measures at least thirty warm Console navigations, separating outgoing teardown, first composer paint, full ready, focus restoration, widget counts by top-level subtree, key-to-echo, and Enter-to-worker scheduling.
2. Establish the baseline distribution and profile subtree weight without changing production code.
3. A/B the Inspector/right-rail content and any measured secondary candidate by suppressing only that subtree in the probe; reject candidates below the 15% first-interactive threshold or outside the full-ready/input budgets.
4. If a candidate clears every threshold, add an ADR and implement the narrow call-after-refresh deferral with RED lifecycle tests; otherwise keep production unchanged.
5. Run fresh-screen, rapid-switch, focus, restore, unmount, and interactive Console soak gates; record median/p95 evidence and close the task honestly.

ADR required: yes
ADR path: `backlog/decisions/078-defer-console-context-rail-content-until-first-refresh.md`
Reason: the first measurement justified evaluating a long-lived lifecycle change; ADR-078 records why the measured candidate was ultimately rejected and production remained eager.

Detailed plan: `Docs/superpowers/plans/2026-08-21-console-context-rail-deferred-mount.md`
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added an isolated Pilot profiler with balanced, rotated control and candidate A/B phases. It records first-refresh interaction, completed full-ready hydration, focus, teardown, input latency, and subtree counts over 30 warm fresh-screen navigations per arm.
- The corrected control run rejected Inspector suppression (11.10% maximum improvement) and advanced Context suppression (21.32%). The real deferred candidate reduced median first interaction 18.31% and first-frame widgets from 401 to 293 while keeping full-ready within budget (+0.13%), but regressed Enter-to-worker p95 12.39%.
- Rejected the candidate under ADR-078 and removed all experimental behavior from production. `ChatScreen`, `ConsoleLeftRail`, and their UI tests are unchanged from the eager implementation; the experiment survives only as profiler-local instrumentation.
- Raw 30-sample reports and summaries live in `Docs/superpowers/qa/console-mount-2026-08-21/`. The canonical nine-route freeze-incident interactivity soak plus fresh-screen, rail, and focus-restore gates passed (13 tests). The profiler's teardown-timestamp regression, Ruff, formatter, compile, and diff checks passed.
- ADR required: yes. ADR-078 records the rejected lifecycle alternative and the measured input-tail reason it was not retained.
<!-- SECTION:NOTES:END -->
