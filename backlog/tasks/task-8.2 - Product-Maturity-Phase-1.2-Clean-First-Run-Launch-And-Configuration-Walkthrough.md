---
id: TASK-8.2
title: >-
  Product Maturity Phase 1.2: Clean First-Run Launch And Configuration
  Walkthrough
status: In Progress
assignee: []
created_date: '2026-05-05 16:26'
updated_date: '2026-05-05 16:50'
labels:
  - product-maturity
  - phase-1-qa-baseline
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-05-product-maturity-phase-1-2-first-run-walkthrough.md
parent_task_id: TASK-8
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify a clean first-run launch and setup-orientation path against the running app so Phase 1 can distinguish usable onboarding from render-only shell coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fresh HOME and XDG walkthrough evidence records launch path and setup state.
- [ ] #2 Home Console Library Settings and recovery entry points are checked from first-run context.
- [ ] #3 Keyboard and visual notes record whether first-run orientation is usable not merely rendered.
- [ ] #4 Any P0 or P1 first-run findings are fixed or explicitly accepted under the product-maturity severity policy.
- [ ] #5 Focused regression evidence protects the first-run launch or evidence-tracking seam changed by this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing product-maturity first-run regression and evidence contract tests.
2. Implement the Textual first-run pilot walkthrough with fresh HOME/XDG setup and setup-entry assertions.
3. Create Phase 1.2 QA evidence from the walkthrough result using the Phase 1 template.
4. Update tracker and Phase 1 README with Phase 1.2 evidence links while keeping Phase 1 open.
5. Run focused verification and close only TASK-8.2 if the evidence is complete.
<!-- SECTION:PLAN:END -->
