---
id: TASK-15507
title: Strengthen high-stakes File Notes state signaling
status: Done
assignee: []
created_date: '2026-08-11 22:51'
updated_date: '2026-08-11 22:59'
labels:
  - notes
  - filesystem
  - accessibility
  - ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-08-11T20-58-28Z__ok-widgets-library-library-file-notes-workspace-py.md
  - backlog/decisions/011-chatbook-workbench-ui-system.md
modified_files:
  - tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
  - Tests/UI/test_library_file_notes_workspace.py
  - >-
    backlog/tasks/task-15507 -
    Strengthen-high-stakes-File-Notes-state-signaling.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make offline linked roots and save conflicts or failures scan as high-stakes states without relying on muted copy alone, while preserving explicit text, compact geometry, and the restrained File Notes visual system.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Offline linked roots remain visibly labeled Offline even when a separate runtime warning is also present.
- [x] #2 Conflict and save-failure status lines receive distinct text-backed semantic state classes that clear when the state changes.
- [x] #3 Warning, offline, conflict, and error treatments use semantic theme tokens without animation, decorative borders, layout shifts, or color-only meaning.
- [x] #4 Rendered status copy remains legible in representative dark, light, and high-contrast themes at normal and 40x20 layouts.
- [x] #5 Focused File Notes state and layout regressions, lint, compile, CSS parsing, and diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add mounted red regressions for offline-plus-warning copy, save-state class transitions, semantic styling, rendered contrast, and unchanged compact geometry.

2. Project explicit offline, warning, conflict, and error classes from the existing root and save state owners while preserving readable text as the primary meaning carrier.

3. Apply restrained semantic background tints and bold text without borders, animation, padding, or size changes.

4. Run focused normal and 40x20 compositor checks across dark, light, and high-contrast themes plus surrounding workspace tests, Ruff, compileall, and diff checks.

ADR required: no

ADR path: backlog/decisions/011-chatbook-workbench-ui-system.md

Reason: This is a routine visual-state and accessibility correction within ADR-011 and DESIGN.md's established semantic-state system. It changes no storage, ownership, sync, security, service, or long-lived application boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added explicit offline, warning, conflict, and error presentation classes to the existing File Notes root and save-state owners. Offline remains in the visible root summary when a separate recovery warning is present.

Applied restrained semantic warning and error background tints with readable theme text and bold weight. The treatments add no border, padding, animation, or geometry-changing decoration, and every state remains fully text-labeled.

Added mounted state-transition regressions plus compositor checks at 120x40 and 40x20 across textual-dark, textual-light, and high_contrast_yellow_black. Painted Offline, Conflict, and Save failed copy measured at least 4.5:1; the focused root, save, conflict-recovery, and disclosed-action matrix passed 13 tests.

Ruff, compileall, and git diff --check passed. CSS bundle generation was not applicable because these scoped rules live in the widget's DEFAULT_CSS and were parsed and painted by the mounted CSS-true tests. No new lesson was required.

ADR required: no. ADR-011 and DESIGN.md already define the applicable semantic-state, text-label, stable-composition, and responsive behavior contract.
<!-- SECTION:NOTES:END -->
