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

2026-09-08 rebase reconciliation plan:
1. Retain dev's Wave-7 LibraryMediaState/LibraryMediaController ownership and its deliberately screen-resident analysis seams. The older pure Analysis extraction is superseded, not layered over that state.
2. Preserve analysis persistence characterization against the current Screen owner, including dev's successful-save row reprojection; retain a live framework-property check against MediaController.
3. Reconcile later forwarding/assembly commits without restoring the obsolete Analysis owner, then run the complete affected media/import/wiring tests. Keep all existing numeric ceilings unchanged.
ADR required: no
ADR path: N/A
Reason: Reconcile a superseded pure move with dev's existing Wave-7 ownership, documented in backlog/docs/library-decomposition-recipe.md section 22; no new boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Extracted media analysis generation, saving, bulk partition/retry and receipt operations into LibraryMediaAnalysisController with explicit late-bound sibling ports. Analysis receipt/edit state is controller-owned; the shared in-flight flag remains on the screen because Import reads it. Provider dispatch and readiness rendering stay screen-resident; private tests patch each owning module. DOM and handler decorators are unchanged.
Verification: 31 analysis characterization tests passed before and after; 142 targeted controller/media/import/module-ratchet tests passed after. New module Ruff and formatter checks plus git diff --check pass. Existing screen size/method ceilings require the planned cleanup and subsequent reader extraction; task remains In Progress pending those.
ADR: no new ADR; applies approved screen decomposition design and DESIGN.md section 7.

2026-09-08 ruling: the preceding extraction notes describe the historical pre-rebase checkpoint. Dev 0fa35d00e8 now owns Media state centrally and six of these methods on MediaController; replaying the old controller would create competing state and method owners. Retain dev's structure instead and preserve the useful behavior checks. Original extraction remains recoverable in codex/pr2427-before-rebase-20260908 (908612df15). Rebased verification remains pending; this does not mark the task Done.
<!-- SECTION:NOTES:END -->
