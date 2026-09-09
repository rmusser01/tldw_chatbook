---
id: TASK-32124
title: >-
  Library Notes Undo after delete restores the database row and the rail count but the row never returns to the tree
status: To Do
assignee: []
created_date: '2026-09-08 21:39'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed once by the evidence assessor: after Undo the count returned to Notes (11) and `deleted=0` in the DB, but the tree still listed 10 titles and Clear filter did not bring the row back. Not re-verified by the parent because the Undo button was unreachable (task-32123). Cause INFERRED: `_undo_library_note_delete` appends the restored record to the flat source records and re-syncs the canvas, while the tree projection is built from paged branch state that is not invalidated. The guide promises that Undo 'immediately returns its row and the Notes rail count'. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After Undo the restored note's row appears in its folder (or Unfiled) without any further action, and focus lands on it
- [ ] #2 Covered by a test through the tree projection, not only the flat list
- [ ] #3 Verified live on a seeded profile
<!-- AC:END -->
