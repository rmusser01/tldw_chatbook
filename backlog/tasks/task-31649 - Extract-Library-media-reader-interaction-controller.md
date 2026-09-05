---
id: TASK-31649
title: Extract Library media reader interaction controller
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 16:55'
updated_date: '2026-09-05 16:56'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move media reading interactions and their transient state into a cohesive controller, restoring Library size and method ratchets while preserving existing Reader behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Content search, reading position, display memoization and read-later behavior preserve existing contracts.
- [ ] #2 Controller dependencies are explicit and late-bound and DOM identities remain unchanged.
- [ ] #3 Targeted Reader characterization and existing unchanged screen size and method ceilings pass.
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

2026-09-08 rebase reconciliation plan:
1. Reconcile this older pure move with dev's Wave-7 MediaState/MediaController, retaining current upstream ownership and the newer reading-position restoration fix.
2. Preserve the progress-drain coalescing/durability characterization against its current owner; omit compatibility scaffolding with no remaining production consumer.
3. Reconcile later receiver-only edits and verify complete Reader/media wiring and affected UI files, without increasing numeric ceilings.
ADR required: no
ADR path: N/A
Reason: Apply the existing Wave-7 ownership documented in backlog/docs/library-decomposition-recipe.md section 22; no new architecture boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

The original extraction checkpoint remains recoverable in backup branch codex/pr2427-before-rebase-20260908 (908612df15). During rebase onto dev 0fa35d00e8, the older Reader owner is superseded by dev's canonical Media state/controller and retained Screen seams, rather than recreating duplicate owners. Historical recipe evidence is retained and explicitly labeled. Current-tree verification remains pending; this task is not Done.

Historical close at 0384f137fc recorded 27 AST-identical moved bodies, 68 Reader checks, 35 Library architecture checks and an earlier 142-case combined selection passing. Keep its independent request-release, mounted-readiness and focus-guard repairs; those are not redundant ownership moves. The rebased comparison found 26 Reader bodies equivalent after state spelling normalization and one newer dev scroll-restoration fix, retained unchanged. Fresh Analysis/Reader characterization plus Media wiring passed 16 checks; the historical Done state does not substitute for complete rebased verification.
