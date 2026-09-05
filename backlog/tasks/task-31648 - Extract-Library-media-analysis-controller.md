---
id: TASK-31648
title: Extract Library media analysis controller
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 16:44'
updated_date: '2026-09-05 17:47'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore Library screen size governance by moving media analysis ownership into its named controller while preserving Reader and Import behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Media analysis provider gates, generation, persistence, overwrite choices and receipts preserve existing behavior.
- [x] #2 Controller dependencies are explicit and late bound; moved bodies preserve their behavior.
- [x] #3 Targeted characterization, architecture and static checks pass without increased existing ceilings.
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
Extracted media analysis generation, saving, bulk partition/retry and receipt operations into LibraryMediaAnalysisController with explicit late-bound sibling ports. Analysis receipt/edit state is controller-owned; the shared in-flight flag remains on the screen because Import reads it. Provider dispatch/readiness rendering and DOM handlers remain screen-resident. Exact state descriptors preserve old assignment seams without broad proxies.
Verification: 31 analysis characterization tests passed before and after; 142 targeted controller/media/import/module-ratchet tests passed; subsequent combined Reader selection passed68 and Library architecture selection passed35. All20 moved analysis bodies were AST-compared unchanged. New modules/tests Ruff and formatter checks plus diffcheck pass; existing screen-wide formatter debt is outside this scoped extraction. Parent reviewed follow-up cleanup. New controller868lines; combined final screen41324lines/1301methods, below unchanged ceilings.
ADR: no new ADR; applies approved screen decomposition design and DESIGN.md section7. Ownership/count details documented in Library decomposition recipe.
<!-- SECTION:NOTES:END -->
