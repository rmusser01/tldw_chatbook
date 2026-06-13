---
id: TASK-6.5
title: Phase 5.5 Replay capability recovery maturity gate
status: Done
assignee:
  - '@codex'
created_date: '2026-05-05 05:22'
updated_date: '2026-05-05 05:25'
labels:
  - unified-shell
  - phase-5
  - recovery
  - qa
  - closeout
dependencies: []
documentation:
  - >-
    Docs/superpowers/qa/unified-shell/phase-5/2026-05-05-shared-recovery-taxonomy.md
  - Docs/superpowers/trackers/unified-shell-maturity-roadmap.md
  - >-
    Docs/superpowers/qa/unified-shell/phase-5/2026-05-05-phase-5-capability-recovery-closeout.md
parent_task_id: TASK-6
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replay Phase 5 capability and recovery workflows in the running app and decide whether the recovery system can be verified or must remain blocked by understandable-recovery gaps.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 running-app QA walkthrough verifies shared recovery patterns across destination blockers runtime-policy denials and optional-dependency blockers
- [x] #2 QA evidence proves blocked states explain what is unavailable why it is unavailable what to do next recovery target and authority owner
- [x] #3 Phase 5 roadmap README and parent task status are updated from the walkthrough result
- [x] #4 Automated closeout regression prevents Phase 5 from being marked verified without durable evidence
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-read Phase 5 child evidence and recovery taxonomy tests to define the maturity-gate checklist.
2. Add a failing closeout regression that expects Phase 5 closeout evidence task state README and roadmap status.
3. Run the focused regression and confirm it fails before closeout updates.
4. Replay focused running-app recovery tests for destination runtime-policy and optional-dependency blocker states.
5. Record Phase 5 closeout QA evidence and update README roadmap and Backlog task state according to the replay result.
6. Run focused closeout recovery tests py_compile where relevant and diff hygiene before committing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added the Phase 5 closeout contract and durable QA artifact for capability recovery. Replayed destination blocker runtime-policy and optional-dependency recovery tests, updated the Phase 5 README and maturity roadmap to verified, and closed the parent Phase 5 task with residual risks documented for live server/auth and full optional-extra execution paths.
<!-- SECTION:NOTES:END -->
