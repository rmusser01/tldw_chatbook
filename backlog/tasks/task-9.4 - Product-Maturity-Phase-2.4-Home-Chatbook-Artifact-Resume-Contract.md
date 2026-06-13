---
id: TASK-9.4
title: 'Product Maturity Phase 2.4: Home Chatbook Artifact Resume Contract'
status: Done
assignee: []
created_date: '2026-05-05 23:14'
updated_date: '2026-05-06 00:05'
labels:
  - product-maturity
  - phase-2-core-agentic-loop
dependencies: []
parent_task_id: TASK-9
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prove Home can surface a recently saved Console Chatbook artifact as resumable work and route it back to Artifacts or Console without manual state reconstruction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Home active-work input includes the latest Console-saved Chatbook artifact when the local Chatbook service has one.
- [x] #2 Home renders the saved artifact as resumable work with Open details and Open in Console controls.
- [x] #3 Home Open details routes to Artifacts for the saved Chatbook artifact.
- [x] #4 Home Open in Console launches the saved artifact with Chatbook id, record id, and Console provenance.
- [x] #5 Missing or failing local Chatbook service remains fail-closed without hiding existing notification or W+C Home behavior.
- [x] #6 QA evidence and product-maturity tracker record the Phase 2.4 exit decision and remaining Phase 2 risk.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing adapter and app-wiring regressions for Home surfacing the latest Console-saved Chatbook artifact.
2. Extend the Home active-work adapter to accept the local Chatbook service and map the latest Console-saved artifact into a Home active-work item.
3. Add handled Home detail and Console control responses for local Chatbook artifact targets.
4. Wire the app Home adapter to the existing local Chatbook service.
5. Record Phase 2.4 QA evidence and update the Phase 2 roadmap tracking.
6. Run focused Home adapter, Home screen, app wiring, and product-maturity tracker verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Home resume support for the latest Console-saved Chatbook artifact by adding a synchronous local Chatbook Home snapshot, mapping the latest artifact into Home active work, routing details to Artifacts, and launching Console with bounded provenance payloads. Added focused Chatbook service, Home adapter, Home screen, app-wiring, tracker, and QA evidence tests. Verification: 49 focused tests passed with 8 warnings.

Review fix pass addressed PR #258 comments: Home now refreshes the Chatbook artifact snapshot off the compose path, mixed W+C plus Chatbook active work exposes dedicated Chatbook controls, and Console launch payload fields are sanitized, path-validated, and bounded before rendering. Verification: 65 focused tests passed with 8 warnings.
<!-- SECTION:NOTES:END -->
