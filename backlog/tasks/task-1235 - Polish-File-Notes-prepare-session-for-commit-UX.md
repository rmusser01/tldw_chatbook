---
id: TASK-1235
title: Polish File Notes prepare-session-for-commit UX
status: In Progress
assignee: []
created_date: '2026-07-28 15:51'
updated_date: '2026-07-28 16:34'
labels:
  - notes
  - git
  - library
  - ux
  - accessibility
dependencies:
  - TASK-1213
references:
  - >-
    .impeccable/critique/2026-07-28T15-38-30Z__ok-widgets-library-library-file-notes-git-panel-py.md
documentation:
  - Docs/superpowers/specs/2026-07-28-file-notes-prepare-session-ux-design.md
  - Docs/superpowers/plans/2026-07-28-file-notes-prepare-session-ux.md
  - backlog/decisions/035-file-notes-session-git-index-controls.md
  - backlog/decisions/033-application-session-state-ownership.md
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - backlog/decisions/029-file-notes-disk-authority.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the shipped File Notes session-staging surface legible, current, note-centered, and usable at narrow terminal sizes while preserving the exact session-only Git safety and ownership behavior delivered by TASK-1213.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The staging surface is titled Prepare session for commit, clearly retains its session-path and complete-file-state scope, and introduces no commit, push, remote, full-repository, or hunk-staging controls.
- [ ] #2 Every session row leads with the note change intent such as Created, Edited, Moved, Deleted, or Restored and separately communicates its actionable Git state and any blocking reason without relying on color alone.
- [ ] #3 Focused Back, Refresh, selected-note, and bulk controls retain readable labels at supported narrow and wide terminal widths, and the visible keyboard guidance accurately describes selection, action, activation, and return behavior.
- [ ] #4 Untrusted, selected-root-changed, repository-changed, unavailable, and non-repository states clear prior rows; checking, stale, or error rows are retained only when the process owner proves the same selected root and complete repository identity.
- [ ] #5 Current repository freshness remains visible independently from the latest action result; a later session or authority generation removes an obsolete action summary, while the action postflight refresh preserves a still-current result and every blocked or error state gives an exact recovery action.
- [ ] #6 A certain successful Stage or Unstage combines the affected session-note count with the checked promise that Chatbook targeted only eligible session paths or restored only its owned session entries; Stage all and Unstage all show their own eligible counts, while non-success, uncertain, mismatched, or zero-effect results omit the promise and retain accurate skipped, clean, or blocked counts.
- [ ] #7 While Prepare session for commit is active beside the editor, editor actions are visually quiet or collapsed and restore on return; narrow editor layouts keep every action label legible without clipping.
- [ ] #8 Focused Textual tests and live acceptance checks at 150×42, 70×28/24, and 40×20 cover authority transitions, focus labels, keyboard return, row semantics, fixed feedback visibility, editor-action quieting, unclipped editor labels, and unrelated-index preservation without adding a full-suite or broad CI gate.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: presentation-only repair conforming to ADR-035, ADR-033, ADR-011, and ADR-029.

1. Add failing focused tests for note-intent rows, semantic/recovery copy, authority clearing, focus, elision, counts, and fixed-region geometry; implement the minimal panel projections and verify green.
2. Add failing focused tests for keyed last-action feedback, promise/count truthfulness, root/repository/session invalidation, wide editor-action quieting, and narrow label fit; implement the minimal workspace presentation changes and verify green.
3. Run the focused UI file, compile, Ruff lint/format checks, bounded live terminal UAT at the approved sizes, and focused review; resolve concrete findings before closeout.

Detailed plan: Docs/superpowers/plans/2026-07-28-file-notes-prepare-session-ux.md
<!-- SECTION:PLAN:END -->
