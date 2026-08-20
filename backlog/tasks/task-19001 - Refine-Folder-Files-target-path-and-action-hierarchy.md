---
id: TASK-19001
title: Refine Folder Files target path and action hierarchy
status: To Do
assignee: []
created_date: '2026-08-20 07:40'
labels:
  - notes
  - files
  - ux
  - accessibility
dependencies:
  - TASK-19000
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce competing controls in Folder files while preserving disk authority, existing guarded file operations, and compact-terminal reachability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The path input has the persistent label `Target path · New / Move / Save copy` in every editor state.
- [ ] #2 The primary action row shows normal or state-critical actions only; secondary actions remain reachable through a keyboard-operable disclosure.
- [ ] #3 Dirty, Conflict, Error, and deleted-file states expose the correct recovery action without hiding Save copy.
- [ ] #4 New and Delete remain spatially separated and every existing guarded file operation keeps its current semantics.
- [ ] #5 The path field and disclosed actions remain focusable, contained, and readable in wide and 40-column compact layouts.
<!-- AC:END -->

## Decision Record Check

ADR required: no new ADR
ADR paths: `backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md`, `backlog/decisions/029-local-private-data-boundary.md`, `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`
Reason: this is presentation-only refinement of the existing disk-authoritative File Notes operations.
