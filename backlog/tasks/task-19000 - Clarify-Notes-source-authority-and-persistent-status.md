---
id: TASK-19000
title: Clarify Notes source authority and persistent status
status: Done
assignee:
  - '@codex'
created_date: '2026-08-20 07:40'
updated_date: '2026-08-20 19:42'
labels:
  - notes
  - ux
  - accessibility
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the two Notes storage modes immediately understandable and keep selected-source authority plus durable operation status visible across Library navigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The source strip reads `Library notes | Folder files` and preserves the existing two-mode routing and storage authorities.
- [x] #2 Every subview shows a pinned, product-language authority row whose currently available operation status survives in-surface canvas navigation without adding legacy persistence.
- [x] #3 Every non-ready state uses text plus a next action; disabled and error states meet the project contrast floor without color-only meaning.
- [x] #4 Library Notes and Folder Files remain keyboard reachable and readable at the supported 60x20 Notes layout.
- [x] #5 The legacy Sync and Import entries remain visible and operable until the atomic cutover task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED source-strip and pinned authority-row tests for Library Notes and Folder Files.\n2. Implement the smallest authority projections at existing compose/update choke points without changing storage or routing.\n3. Add RED compact-navigation and readable-error checks, then regenerate CSS.\n4. Run focused UI/CSS gates, perform spec and quality review, update documentation and task evidence.\n\nADR required: no new ADR\nADR path: backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md; backlog/decisions/029-local-private-data-boundary.md; backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md\nReason: presentation, copy, contrast, and retained-status refinement only; existing authority and behavior remain unchanged.\n\nPlan: Docs/superpowers/plans/2026-08-20-notes-files-presentation-refinement.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the approved Notes authority presentation without changing routing, storage, sync ownership, or legacy entry points. Renamed the source strip to Library notes | Folder files; added retained authority rows for Library Notes and Folder Files; centralized Folder Files into a bounded two-line projection that preserves folder identity, root/save/Git/push state, and an honest next action at 60x20; added honest checking, root-change, file-operation, and running-notes states; and updated the Library user guides. Added mounted production-CSS compositor coverage plus a 1,080-combination authority-state invariant. Existing ADR-021, ADR-029, and ADR-031 govern the unchanged boundaries; no new ADR was required. Verification: focused final gate 15 passed; full Notes/File owning suites 125 passed; shell/honesty/accessibility/CSS 633 passed; Ruff, CSS bundle parity, and git diff --check passed. Independent spec and quality review approved with no remaining findings.
<!-- SECTION:NOTES:END -->
