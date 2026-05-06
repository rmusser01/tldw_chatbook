---
id: TASK-10.5
title: 'Product Maturity Phase 3.5: Core Product Loop Screen Adaptation'
status: Done
assignee: []
created_date: '2026-05-06 13:25'
updated_date: '2026-05-06 13:25'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - gate-1-core-loop
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md
  - Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md
  - Docs/superpowers/plans/2026-05-06-gate-1-core-product-loop-screen-adaptation.md
parent_task_id: TASK-10
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Adapt the Home Console and Library screens into the approved Gate 1 core product-loop layout so users can understand status stage or control work and act on Library sources without relying on legacy route knowledge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Home exposes dashboard regions and selected-item inspector.
- [x] #2 Console exposes staged context transcript inspector and composer contract regions.
- [x] #3 Library modes are actionable without leaving Library.
- [x] #4 Mounted UI regressions cover Home Console and Library.
- [x] #5 QA walkthrough verifies the app is usable not merely clickable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing mounted Gate 1 regressions for Home Console and Library.
2. Adapt Home into Command Center dashboard regions while preserving runtime controls.
3. Adapt Console into Agent Workbench shell regions around the existing chat surface.
4. Make Library mode chips actionable with mode-specific detail copy.
5. Record QA evidence roadmap links and Backlog task hygiene.
6. Run focused verification and diff hygiene before PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Gate 1 core-loop screen adaptation. Home now renders command-center dashboard regions with a selected-item inspector while existing operational controls still route through the app hooks. Console now wraps the existing chat surface in agentic shell regions for staged context transcript run inspector and composer contracts while preserving live-work readiness and pending launch behavior. Library modes are now clickable and update detail copy without leaving Library or breaking source snapshot Study or Console handoff paths. Added mounted regressions plus QA evidence roadmap README and Backlog tracking.
<!-- SECTION:NOTES:END -->
