---
id: TASK-8.7
title: 'Product Maturity Phase 1.7: Narrow Core Loop Proof'
status: Done
assignee: []
created_date: '2026-05-05 20:19'
updated_date: '2026-05-05 20:26'
labels:
  - product-maturity
  - phase-1-qa-baseline
dependencies:
  - TASK-8.6
parent_task_id: TASK-8
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the remaining Phase 1 gate by proving one source or Search/RAG query can reach Console with staged context and a recoverable output path or honest runtime blocker.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Running-app QA walkthrough verifies a Library or Search/RAG to Console core loop.
- [x] #2 Focused regression coverage proves staged context reaches Console and exposes source authority.
- [x] #3 Repo-tracked QA evidence records automated evidence manual walkthrough residual risk and exit decision.
- [x] #4 P0/P1 findings are fixed or explicitly accepted according to the severity policy.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reuse existing Search/RAG and Console handoff seams to define the narrow core-loop contract.
2. Add a focused product-maturity UI regression proving a RAG result stages context into Console.
3. Record Phase 1.7 QA evidence and update the Phase 1 README and product-maturity roadmap.
4. Mark the Phase 1 parent and Phase 1.7 child complete only if the gate evidence is clean.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a Phase 1.7 running-app regression proving Search/RAG result context stages into Console with local source authority and a review-before-send draft prompt. Added Phase 1.7 QA evidence and updated the product-maturity roadmap and Phase 1 README to close the Phase 1 QA baseline. Full grounded generation and Artifact/Chatbook persistence remain scoped to Phase 2.
<!-- SECTION:NOTES:END -->
