---
id: TASK-7.1
title: Phase 6.1 Replay first-time user walkthrough
status: Done
assignee:
  - '@codex'
created_date: '2026-05-05 05:35'
updated_date: '2026-05-05 05:45'
labels:
  - unified-shell
  - phase-6
  - audit-replay
  - first-time-user
  - qa
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-03-unified-shell-maturity-tracking-design.md
  - Docs/superpowers/qa/unified-shell/phase-1/walkthrough-protocol.md
  - Docs/superpowers/trackers/unified-shell-maturity-roadmap.md
  - Docs/superpowers/qa/unified-shell/phase-6/2026-05-05-phase-6-1-first-time-user-replay.md
parent_task_id: TASK-7
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replay the Unified Shell from a first-time user perspective against the running Textual app so orientation navigation labels recovery states and obvious starting paths are verified with durable evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Clean first-time launch walkthrough records Home Console navigation and at least three top-level destination orientation paths
- [x] #2 Evidence captures visual usability keyboard path functional result defect severity and residual risk using the QA walkthrough protocol
- [x] #3 Findings distinguish first-time onboarding gaps from intentionally deferred service-depth work
- [x] #4 Phase 6 README roadmap and parent task tracking are updated without prematurely closing Phase 6
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing Phase 6 first-time replay contract that expects durable evidence README roadmap and task tracking.
2. Run the focused contract to confirm it fails before evidence exists.
3. Exercise the running Textual app through the app test harness from a clean first-time state, covering Home Console and representative destinations.
4. Record first-time walkthrough evidence with workflow matrix visual usability notes defects residual risks and verification commands.
5. Update the Phase 6 README roadmap and Backlog task state without closing the Phase 6 parent.
6. Run focused Phase 6 tests plus relevant shell navigation and destination checks before committing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a Phase 6.1 first-time replay contract and durable QA evidence for clean Home launch Console orientation and representative Library Personas and Skills orientation paths. Updated the Phase 6 README roadmap and parent task to in-progress while leaving power-user and Nielsen closeout criteria open.
<!-- SECTION:NOTES:END -->
