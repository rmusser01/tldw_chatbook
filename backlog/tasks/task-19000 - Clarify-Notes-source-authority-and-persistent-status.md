---
id: TASK-19000
title: Clarify Notes source authority and persistent status
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-20 07:40'
updated_date: '2026-08-20 14:48'
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
- [ ] #1 The source strip reads `Library notes | Folder files` and preserves the existing two-mode routing and storage authorities.
- [ ] #2 Every subview shows a pinned, product-language authority row whose currently available operation status survives in-surface canvas navigation without adding legacy persistence.
- [ ] #3 Every non-ready state uses text plus a next action; disabled and error states meet the project contrast floor without color-only meaning.
- [ ] #4 Library Notes and Folder Files remain keyboard reachable and readable at the supported 60x20 Notes layout.
- [ ] #5 The legacy Sync and Import entries remain visible and operable until the atomic cutover task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED source-strip and pinned authority-row tests for Library Notes and Folder Files.\n2. Implement the smallest authority projections at existing compose/update choke points without changing storage or routing.\n3. Add RED compact-navigation and readable-error checks, then regenerate CSS.\n4. Run focused UI/CSS gates, perform spec and quality review, update documentation and task evidence.\n\nADR required: no new ADR\nADR path: backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md; backlog/decisions/029-local-private-data-boundary.md; backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md\nReason: presentation, copy, contrast, and retained-status refinement only; existing authority and behavior remain unchanged.\n\nPlan: Docs/superpowers/plans/2026-08-20-notes-files-presentation-refinement.md
<!-- SECTION:PLAN:END -->
