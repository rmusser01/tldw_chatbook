---
id: TASK-13.2
title: 'Phase 6.2: Full first-time user release replay'
status: Done
assignee: []
created_date: 2026-05-16 00:00
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
Verify a first-time user can launch the running app, understand the product model, diagnose setup blockers, and find the correct starting workflows without developer knowledge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 QA walkthrough verifies clean first-run launch, Home orientation, Console orientation, Library entry, and Settings/setup recovery in the running app.
- [x] #2 Focused regression evidence exists for first-time launch, navigation, and setup-orientation seams.
- [x] #3 Repo-tracked QA evidence exists under Docs/superpowers/qa/product-maturity/phase-6/.
- [x] #4 P0/P1 findings are fixed or explicitly accepted according to the spec severity policy.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing Phase 6.2 release replay evidence/tracking regression.
2. Replay clean first-run launch in a mounted Textual app with clean HOME/XDG state, covering Home, Console, Library, and Settings/setup recovery.
3. Document QA evidence, update the Phase 6 QA index and product maturity roadmap, and close TASK-13.2 only if P0/P1 findings are fixed or accepted.
4. Run focused release replay and first-run/navigation regressions plus diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replayed the Phase 6.2 first-time release path in the mounted Textual app with clean HOME/XDG environment state. Verified first-run routing to Home, Console setup recovery, Library entry points, and Settings orientation without finding P0/P1 blockers. Added repo-tracked QA evidence, updated the Phase 6 QA index and product-maturity roadmap, and added a focused regression covering both the running-app replay and evidence/tracking contract.
<!-- SECTION:NOTES:END -->
