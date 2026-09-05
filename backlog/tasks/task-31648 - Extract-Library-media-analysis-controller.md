---
id: TASK-31648
title: Extract Library media analysis controller
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 16:44'
updated_date: '2026-09-05 16:55'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore Library screen size governance by moving media analysis ownership into its named controller while preserving Reader and Import behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Media analysis provider gates, generation, persistence, overwrite choices and receipts preserve existing behavior.
- [ ] #2 Controller dependencies are explicit and late bound; moved bodies preserve their behavior.
- [ ] #3 Targeted characterization, architecture and static checks pass without increased existing ceilings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize Reader generation and bulk/import analysis using existing media and import tests.
2. Extract one media analysis controller with explicit late-bound dependencies, controller-owned state and stable screen entry points. Preserve method bodies and DOM.
3. Verify targeted tests, new wiring checks, Ruff, formatter and diff; measure and pin the new controller.
ADR required: no
ADR path: N/A
Reason: Direct application of approved Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md and DESIGN.md section 7; no boundary or behavior redesign.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Extracted media analysis generation, saving, bulk partition/retry and receipt operations into LibraryMediaAnalysisController with explicit late-bound sibling ports. Analysis receipt/edit state is controller-owned; the shared in-flight flag remains on the screen because Import reads it. Provider dispatch and readiness rendering stay screen-resident; private tests patch each owning module. DOM and handler decorators are unchanged.
Verification: 31 analysis characterization tests passed before and after; 142 targeted controller/media/import/module-ratchet tests passed after. New module Ruff and formatter checks plus git diff --check pass. Existing screen size/method ceilings require the planned cleanup and subsequent reader extraction; task remains In Progress pending those.
ADR: no new ADR; applies approved screen decomposition design and DESIGN.md section 7.
<!-- SECTION:NOTES:END -->
