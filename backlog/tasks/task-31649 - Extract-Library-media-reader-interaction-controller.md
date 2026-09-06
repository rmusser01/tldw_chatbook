---
id: TASK-31649
title: Extract Library media reader interaction controller
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 16:55'
updated_date: '2026-09-05 17:47'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move media reading interactions and their transient state into a cohesive controller, restoring Library size and method ratchets while preserving existing Reader behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Content search, reading position, display memoization and read-later behavior preserve existing contracts.
- [x] #2 Controller dependencies are explicit and late-bound and DOM identities remain unchanged.
- [x] #3 Targeted Reader characterization and existing unchanged screen size and method ceilings pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run existing Reader search, progress, memoization and read-later characterization before extraction.
2. Extract one Reader interaction controller owning its search/progress/memo state; retain DOM structure and explicit screen callbacks.
3. Remove proven-obsolete private delegators and use exact per-field forwarding declarations for transitional state, mirroring the existing Console descriptor.
4. Verify targeted Reader/media/import tests, new controller ports, unchanged architecture ceilings, Ruff/format and diff checks.
ADR required: no
ADR path: N/A
Reason: Direct application of approved screen decomposition design and DESIGN.md section 7; state forwarding mirrors the existing Console convention.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Extracted content search, display-state memoization, coalesced reading-position writes and read-later interactions into LibraryMediaReaderController. Named late-bound ports retain shared Reader session/detail ownership; exact per-field descriptors preserve private state assignments. Retired obsolete delegators and migrated private callers, retaining actual DOM/action handlers and the shared canvas analysis-reason seam. All27 moved Reader method bodies were AST-compared unchanged.
Verification: exact five-file Reader selection68passed181.50s; Library screen/module ratchet selection35passed3deselected3.36s; previous combined media/import/controller selection142passed. Existing baseline failures were separately repaired: await final debounced fake requests before release, await mounted Reader controls, and release superseded Media entry-focus guards. No assertion meanings or timeout ceilings changed. New modules/tests Ruff+format and scoped legacy-test E9/F checks pass, diffcheckclean. Existing whole-screen formatter debt was not bulk-reformatted. Parent reviewed runtime cleanup with no actionable finding.
Controller766lines; extraction lowered screen42558/1319→41325/1301; separate runtime cleanup gives41324/1301. Recipe updated. ADR: no new ADR; direct application of approved screen decomposition design and DESIGN.md section7.
<!-- SECTION:NOTES:END -->
