---
id: TASK-31560
title: Complete new Library modal contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 01:41'
updated_date: '2026-09-05 01:46'
labels:
  - library
  - tests
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore bidirectional Library modal inventory and safe-dismissal coverage for the shipped multi-skill import choice and review-set picker modals.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The production presenter edges resolve to `SkillImportChoiceModal` and `LibraryReviewSetPickerDialog`.
- [x] #2 Both modals have negative, positive, focus, and lifecycle contract coverage.
- [x] #3 Skill import choice initializes the shared dismissal mixin exactly once.
- [x] #4 The Library modal inventory and focused contract tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce each unresolved constructor and confirm the AST scanner sees the presenters' local imports but lacks the concrete types in its contract table.
2. Add both modals' exact factories, dismissal/result contracts, presenter edges, and public positive drivers; remove any declared edge whose wrapper now only delegates and constructs no modal.
3. Remove the choice modal's duplicate explicit mixin mount call exposed by lifecycle coverage.
4. Run the bidirectional inventory plus both modals' generated contract cases, Ruff, and diff checks.

ADR required: no
ADR path: N/A
Reason: this extends test coverage for an already-shipped modal and does not change the modal or its ownership boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added exact contract-table entries, production presenter edges, negative/positive result oracles, and public positive drivers for both newly shipped Library modals.
- Removed the stale LibraryScreen export edge now owned by its controller and removed `SkillImportChoiceModal`'s explicit `super().on_mount()` call, which caused the shared dismissal mixin to initialize twice under Textual's handler dispatch.
- Evidence: the 15 focused new/inventory cases pass and the complete Library modal ownership module passes 182/182.
- ADR required: no; the modal behavior and presenter boundaries already existed.
<!-- SECTION:NOTES:END -->
