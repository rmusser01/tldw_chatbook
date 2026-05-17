---
id: TASK-13.3
title: 'Phase 6.3: Power-user workflow release replay'
status: Done
assignee: []
created_date: '2026-05-16 00:00'
labels:
  - product-maturity
  - phase-6-release-hardening
dependencies:
  - TASK-13.1
parent_task_id: TASK-13
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replay at least five core workflows from a power-user perspective and verify repeated use remains fast, visible, and recoverable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 QA walkthrough verifies at least five running-app workflows across grounded answer, source-to-artifact, agent run, monitoring, study, and recovery loops.
- [x] #2 Focused regression evidence exists for the changed or replayed workflow seams.
- [x] #3 Repo-tracked QA evidence exists under Docs/superpowers/qa/product-maturity/phase-6/.
- [x] #4 P0/P1 findings are fixed or explicitly accepted according to the spec severity policy.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing Phase 6.3 mounted replay and evidence/tracking regression.
2. Replay returning-user Home to Console, Library Search/RAG mode, Import/Export, study affordances, and Watchlists live-work recovery/follow-through.
3. Document the workflow matrix, P0/P1 disposition, screenshot-gate decision, and residual risks.
4. Update the Phase 6 QA index and product-maturity roadmap.
5. Run focused Phase 6.3 replay verification and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified the power-user release replay in the running mounted app without changing visible UI. The replay covers six workflows: grounded answer, source-to-artifact, agent run, monitoring, study, and recovery. Added durable Phase 6.3 QA evidence with a workflow matrix, updated the QA index and product-maturity roadmap, and added focused regression coverage for the repeated-use routing and evidence contract. No P0/P1 blockers were found.
<!-- SECTION:NOTES:END -->
