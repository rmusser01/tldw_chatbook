---
id: TASK-22032
title: Migrate Library Notes to the adaptive reader shell
status: Done
assignee:
  - '@codex'
created_date: '2026-08-24 23:25'
updated_date: '2026-08-25 17:07'
labels:
  - library
  - ui
dependencies:
  - TASK-22031
references:
  - >-
    Docs/superpowers/specs/2026-08-24-library-destinations-adaptive-reader-design.md
  - backlog/decisions/086-library-adaptive-reader-shell.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move Database Notes into the shared Library adaptive reader structure while preserving the existing editor coordinator templates import sync conflict recovery utilities and destructive-action contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Notes list remains mounted beside a permanent work pane with independent list collapse and destination-specific geometry
- [x] #2 Edit is the default mode and mounting it does not mark the note dirty
- [x] #3 Edit Preview and Info share the current item-owned draft and Preview renders unsaved draft content
- [x] #4 Create templates import sync conflicts recovery utilities and destructive actions remain reachable without unmounting the list
- [x] #5 Selection loading dirty-draft navigation stale workers deletion and retry follow the approved identity and recovery contracts
- [x] #6 No multi-item draft registry or new Notes authority is introduced
- [x] #7 Automated list editor conflict geometry focus and capability tests pass with a representative live TUI walkthrough
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory Notes capabilities and draft authority
2. Add presentation-only reader state with test-first identity and mode contracts
3. Split the persistent list and permanent work pane without changing Notes authority
4. Verify workflows, geometry, focus, and capability preservation

ADR required: yes
ADR path: backlog/decisions/086-library-adaptive-reader-shell.md
Reason: consumes the accepted Library structural boundary without changing Notes authority.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Mounted Database Notes as one shared adaptive reader with retained Library, Items, and Work owners. The Notes list remains mounted while the work pane projects loading, Edit, Preview, Info, Create/templates, import, lasting sync, conflict, delete, and recovery states.
- Kept `LibraryNoteSessionCoordinator` and its snapshot as the sole item-draft authority. Presentation state only controls mode, compact layout, bulk read-only labelling, and retained-pane visibility; no multi-item draft registry or replacement authority was added.
- Preserved the existing dirty-flush, generation, version, destructive-admission, retry, exact-placement, and receipt contracts. Final review hardening added dirty guards to permanent Navigator tasks, delayed placement identity commits until navigation is permitted, and prevented Back/Save from discarding or mutating the labelled bulk preview.
- Added adaptive geometry, independent list collapse, compact controls, truthful focus restoration, and narrow-width minimum handling. A live TUI walkthrough at 170x48 confirmed all three panes and at 90x30 confirmed optional navigation collapse with Work receiving the available width.
- Verification after rebasing onto `origin/dev` at `1b21f5339`: 133 Notes shell tests passed; 88 reader/state/widget/CSS tests passed; 29 import/sync/File Notes journey tests passed with one unchanged baseline case deselected; Ruff, compileall, and `git diff --check` passed. A four-file branch-versus-`dev` comparison produced the same 28 pre-existing failures on each side and no branch-only failure. Two review passes ended with no actionable findings.
- ADR: implemented the existing boundary in `backlog/decisions/086-library-adaptive-reader-shell.md`; no new ADR was required because storage, sync ownership, and service contracts were unchanged. No generalisable new repository lesson was identified beyond the task-specific regression coverage added here.
<!-- SECTION:NOTES:END -->
