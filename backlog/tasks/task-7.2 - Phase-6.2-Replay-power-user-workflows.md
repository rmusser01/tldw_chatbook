---
id: TASK-7.2
title: Phase 6.2 Replay power-user workflows
status: Done
assignee:
  - '@codex'
created_date: '2026-05-05 07:20'
updated_date: '2026-05-05 07:20'
labels:
  - unified-shell
  - phase-6
  - audit-replay
  - power-user
  - qa
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-03-unified-shell-maturity-tracking-design.md
  - Docs/superpowers/qa/unified-shell/phase-1/walkthrough-protocol.md
  - Docs/superpowers/trackers/unified-shell-maturity-roadmap.md
  - Docs/superpowers/qa/unified-shell/phase-6/2026-05-05-phase-6-2-power-user-workflow-replay.md
parent_task_id: TASK-7
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replay the Unified Shell from a power-user perspective against the running Textual app so fast repeated workflows Console source readiness Library source actions and live-work follow-through are verified with durable evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Configured Home primary action opens Console and shows live-work source readiness.
- [x] #2 Library Search/RAG and Import/Export workflows are replayed from the running app.
- [x] #3 Console live-work status-card follow-through opens W+C run context.
- [x] #4 Phase 6 README roadmap and parent task tracking are updated without prematurely closing Phase 6.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing Phase 6.2 power-user replay contract that expects durable evidence README roadmap and task tracking.
2. Run the focused contract to confirm it fails before evidence exists.
3. Exercise the running Textual app through the app test harness in a configured returning-user state.
4. Record power-user workflow evidence with workflow matrix repeated-use findings defects residual risks and verification commands.
5. Update the Phase 6 README roadmap and Backlog task state while leaving Nielsen closeout open.
6. Run focused Phase 6 tests plus relevant shell navigation and product-model checks before committing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a Phase 6.2 power-user replay contract and durable QA evidence for configured Home-to-Console launch Console live-work readiness Library Search/RAG Library Import/Export and W+C live-work follow-through. Updated Phase 6 tracking to include TASK-7.2 while keeping the parent phase in progress for Nielsen closeout.
<!-- SECTION:NOTES:END -->
