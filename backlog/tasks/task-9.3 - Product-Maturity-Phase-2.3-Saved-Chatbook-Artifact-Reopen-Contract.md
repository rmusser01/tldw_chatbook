---
id: TASK-9.3
title: 'Product Maturity Phase 2.3: Saved Chatbook Artifact Reopen Contract'
status: Done
assignee: []
created_date: '2026-05-05 22:21'
updated_date: '2026-05-05 22:25'
labels:
  - product-maturity
  - phase-2-core-agentic-loop
dependencies: []
parent_task_id: TASK-9
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prove that a Chatbook artifact record created from Console can be discovered in Artifacts and reopened into Console with saved-response provenance, while leaving Home resume as a separate remaining Phase 2 risk.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Artifacts identifies Console-saved Chatbook artifact records from local Chatbook metadata.
- [x] #2 Artifacts shows saved-response provenance or source authority before launch so the user understands what will reopen.
- [x] #3 Launching the saved artifact routes into Console with record id, Chatbook id, and saved-response provenance sufficient to resume or review without manual state reconstruction.
- [x] #4 Empty-state and service-failure recovery remain honest and existing Artifacts launch behavior does not regress.
- [x] #5 QA evidence and product-maturity tracker record the Phase 2.3 exit decision and remaining Home resume risk.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing Artifacts destination regression for Console-saved Chatbook metadata visibility and Console reopen payload provenance.
2. Implement the smallest safe Artifacts metadata mapping and provenance copy needed to pass the regression.
3. Preserve existing latest-chatbook launch selection, sanitization, empty-state, and service-failure behavior.
4. Record Phase 2.3 QA evidence and update the product-maturity tracker and Phase 2 README.
5. Run focused Artifacts/Console handoff tests, tracker tests, and diff hygiene before closing the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Phase 2.3 saved Chatbook artifact reopen contract. Artifacts now recognizes Console-saved Chatbook metadata, surfaces provider/model provenance and bounded saved-response preview, and carries sanitized provenance into the Console launch payload while preserving existing latest-chatbook, empty-state, service-failure, and sanitization behavior. Added focused Artifacts/Console regression coverage, Phase 2.3 QA evidence, roadmap tracking, and a product-maturity tracker regression.
<!-- SECTION:NOTES:END -->
