---
id: TASK-19001
title: Refine Folder Files target path and action hierarchy
status: Done
assignee:
  - '@codex'
created_date: '2026-08-20 07:40'
updated_date: '2026-08-20 21:03'
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
- [x] #1 The path input has the persistent label `Target path · New / Move / Save copy` in every editor state.
- [x] #2 The primary action row shows normal or state-critical actions only; secondary actions remain reachable through a keyboard-operable disclosure.
- [x] #3 Dirty, Conflict, Error, and deleted-file states expose the correct recovery action without hiding Save copy.
- [x] #4 New and Delete remain spatially separated and every existing guarded file operation keeps its current semantics.
- [x] #5 The path field and disclosed actions remain focusable, contained, and readable in wide and 40-column compact layouts.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED path-label and editor-state action-visibility matrix tests, including the required primary order and recovery promotions.\n2. Implement layout-only hierarchy changes at the existing label, visibility, disclosure, control, and layout choke points while preserving every guarded handler and service call.\n3. Add production-shaped wide and 40-column compositor/focus tests, including focus redirection when secondary actions hide.\n4. Run the full Folder Files suite, static checks, spec/quality review, update user documentation if visible behavior changed, and close the task with exact evidence.\n\nADR required: no new ADR\nADR path: backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md; backlog/decisions/029-local-private-data-boundary.md; backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md\nReason: presentation-only refinement of existing disk-authoritative File Notes operations; no storage, authorization, service, or shortcut contract changes.\n\nPlan: Docs/superpowers/plans/2026-08-20-notes-files-presentation-refinement.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the approved Folder Files action hierarchy without changing storage, routing, guarded file handlers, service calls, messages, or shortcuts. The target input now keeps the persistent label Target path · New / Move / Save copy; normal primary order is New, Move, Delete, More file actions; Restore, Reload, and Save copy/Export exact copy are promoted only when state-critical; Protect, normal Reload, and Refresh remain behind the existing keyboard disclosure. The single Reload control is conditionally reordered so conflict/error place it before Save copy while normal forward Tab traverses More, Reload, Protect, Refresh. Reload confirmation remains Cancel-first and keeps Target path plus Save copy reachable; the editor pane owns compact scrolling and conflict resolution uses scroll-aware safe focus. Updated the File Notes guide and added production-shaped 40-column compositor, keyboard, focus-redirection, confirmation, and state-matrix coverage. Existing ADR-021, ADR-029, and ADR-031 govern the unchanged boundaries; no new ADR was required. Verification: full owning suite 122 passed; root final slice 14 passed; CSS integrity 11 passed; Ruff, CSS bundle parity, and git diff --check passed. Independent spec and quality reviews approved with no remaining findings.
<!-- SECTION:NOTES:END -->
