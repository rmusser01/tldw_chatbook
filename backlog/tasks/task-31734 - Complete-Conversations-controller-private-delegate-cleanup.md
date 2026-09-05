---
id: TASK-31734
title: Complete Conversations controller private delegate cleanup
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:29'
updated_date: '2026-09-05 19:40'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove obsolete private screen forwarding methods after migrating their remaining callers to the existing Conversations owner, preserving behavior and exact test assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Eight private screen methods are removed and their callers use the existing Conversations controller without changing controller bodies
- [x] #2 Existing conversation behavior and explicit pruned-name architecture checks pass with unchanged behavioral assertions
- [x] #3 Library method count meets the existing ceiling without budget relaxation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the eight-name repository census and characterize existing conversations behavior.
2. Retarget callers and the sole direct UI-test invocation, remove the eight pure delegates, and extend the existing pruned-name inventory.
3. Run conversations UI/architecture coverage, scoped static checks, and parent review; record exact counts.

ADR required: no
ADR path: N/A
Reason: Mechanical cleanup follows DESIGN.md section 7 and the approved Library decomposition recipe, preserving existing ownership and contracts.

Detailed plan: `Docs/superpowers/plans/2026-09-05-library-assembly-cleanup.md`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed eight obsolete private screen delegates and migrated their screen callers to the existing Conversations controller. The sole direct UI-test invocation follows the same owner; all behavioral assertion ASTs are unchanged. Controller bodies and all 100 Library diagnostic statements remain unchanged.
Baseline 72 conversation UI tests passed (68.92s); expanded pruned-name check failed before deletion; after-change 72 UI plus 7 architecture checks passed (71.46s). Test Ruff and diffcheck pass. Bounded screen formatting applied; all 40 preexisting screen Ruff findings unchanged. Screen: 41627 lines / 1301 methods, down 24 lines / 8 methods. The separate ordered-assembly task addresses the remaining line ceiling.
ADR required: no; mechanical ownership cleanup under DESIGN.md section 7 and the approved recipe. Parent reviewed the exact diff and approved the scoped commit. No behavior changes or new lessons.
<!-- SECTION:NOTES:END -->
