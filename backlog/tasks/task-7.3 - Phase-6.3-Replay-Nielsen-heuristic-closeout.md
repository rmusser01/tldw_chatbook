---
id: TASK-7.3
title: Phase 6.3 Replay Nielsen heuristic closeout
status: Done
assignee:
  - '@codex'
created_date: '2026-05-05 07:40'
updated_date: '2026-05-05 07:40'
labels:
  - unified-shell
  - phase-6
  - audit-replay
  - nielsen
  - qa
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-03-unified-shell-maturity-tracking-design.md
  - Docs/superpowers/qa/unified-shell/phase-1/walkthrough-protocol.md
  - Docs/superpowers/trackers/unified-shell-maturity-roadmap.md
  - Docs/superpowers/qa/unified-shell/phase-6/2026-05-05-phase-6-3-nielsen-heuristic-closeout.md
parent_task_id: TASK-7
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replay the Unified Shell against Nielsen usability heuristics from a senior UX perspective so remaining shell-level defects residual risks and closeout decisions are documented with durable running-app evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Nielsen heuristic evidence covers all ten heuristics against running-app shell behavior.
- [x] #2 Evidence identifies prioritized residual findings and distinguishes blockers from future service-depth work.
- [x] #3 Phase 6 README roadmap and parent task tracking are updated to verified only after first-time power-user and Nielsen evidence all exist.
- [x] #4 Focused and broader shell replay tests pass after the closeout update.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing Phase 6.3 Nielsen closeout contract that expects durable evidence README roadmap and task tracking.
2. Run the focused contract to confirm it fails before closeout evidence exists.
3. Exercise the running Textual app through the app test harness across Home Console ACP Library and Settings heuristic signals.
4. Record Nielsen heuristic evidence with prioritized findings deferred work residual risks and verification commands.
5. Update the Phase 6 README roadmap parent task and child task state to verified/done.
6. Run focused Phase 6 tests plus relevant shell navigation and product-model checks before committing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a Phase 6.3 Nielsen closeout contract and durable QA evidence mapping all ten heuristics to running-app shell behavior. Updated Phase 6 tracking to verified and closed the parent task while preserving residual risks for Library subroute return affordance full keyboard sweep and future service-depth live paths.
<!-- SECTION:NOTES:END -->
