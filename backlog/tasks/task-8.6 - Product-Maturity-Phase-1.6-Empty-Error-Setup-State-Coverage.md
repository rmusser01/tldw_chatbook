---
id: TASK-8.6
title: 'Product Maturity Phase 1.6: Empty Error Setup State Coverage'
status: Done
assignee: []
created_date: '2026-05-05 19:40'
updated_date: '2026-05-05 19:40'
labels:
  - product-maturity
  - phase-1-qa-baseline
dependencies:
  - TASK-8.5
parent_task_id: TASK-8
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify clean-run empty error and setup states expose honest owner reason impact and recovery actions before Phase 1 closes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Clean-run evidence records missing provider optional dependency service unavailable and local-runtime setup states.
- [x] #2 Each verified blocked or empty state explains owner reason impact and next action without false affordances.
- [x] #3 Any P0/P1 empty error or setup findings are fixed or explicitly accepted under the product-maturity severity policy.
- [x] #4 Focused regression coverage protects the Phase 1.6 evidence tracker README and task closeout seams.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused failing Phase 1.6 contract test for empty/error/setup evidence, tracker, README, and task closeout.
2. Inspect existing clean-run setup and recovery surfaces for provider, optional dependency, service unavailable, and local-runtime blockers.
3. Create Phase 1.6 QA evidence documenting verified blocked states, residual risk, and exit decision.
4. Update the Phase 1 README and product-maturity tracker while keeping Phase 1 open for the core-loop proof gate.
5. Mark TASK-8.6 acceptance criteria complete after focused verification and add implementation notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a Phase 1.6 regression that exercises clean-run setup blockers, Console missing API key recovery, ACP runtime-not-configured recovery, Search/RAG missing optional dependency recovery, and Library/Personas/W+C/Skills service-unavailable states. Added Phase 1.6 QA evidence plus roadmap and Phase 1 README links while keeping Phase 1 open for the narrow core-loop proof gate. No P0/P1 empty/error/setup blockers were found in this slice.
<!-- SECTION:NOTES:END -->
