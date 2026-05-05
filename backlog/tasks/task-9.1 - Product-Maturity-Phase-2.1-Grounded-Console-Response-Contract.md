---
id: TASK-9.1
title: 'Product Maturity Phase 2.1: Grounded Console Response Contract'
status: Done
assignee: []
created_date: '2026-05-05 21:03'
updated_date: '2026-05-05 21:07'
labels:
  - product-maturity
  - phase-2-core-agentic-loop
dependencies: []
parent_task_id: TASK-9
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prove staged Search/RAG context can be sent from Console with model-bound grounded context and honest missing-runtime recovery while deferring Artifact or Chatbook persistence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Running app or focused UI regression verifies staged Search RAG context is sent through Console with source authority preserved
- [x] #2 Model bound prompt or request includes staged evidence metadata and user prompt
- [x] #3 Missing provider or runtime path preserves staged context and exposes recovery instead of silently dropping the source
- [x] #4 Repo tracked QA evidence records automated evidence residual risk and Phase 2.1 exit decision
- [x] #5 P0 and P1 findings are fixed or explicitly accepted according to the severity policy
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current ChatSession send path and handoff payload formatting seams
2. Add failing focused regression for staged context send behavior and missing runtime preservation
3. Implement minimal code to route staged context into the model bound prompt or request or preserve blocker state
4. Add Phase 2.1 QA evidence and update roadmap or task state
5. Verify focused adjacent handoff and relevant Phase 2 and Phase 1 regressions
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Phase 2.1 grounded Console response contract. Added focused regressions proving staged Search/RAG context reaches the provider-bound chat_wrapper message with source authority metadata and proving blocked tab send paths preserve staged context. Updated tab send completion to mark handoff payloads sent only after generation work starts and hardened legacy tab selector handling to ignore non-string mock tab IDs. Added Phase 2 QA evidence and product-maturity tracker updates. Verification covered focused event handler suites, adjacent Chat handoff/core-loop tests, full Phase 1 product-maturity sweep, and git diff whitespace checks.
<!-- SECTION:NOTES:END -->
