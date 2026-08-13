---
id: TASK-15703
title: Render and operate the Database Notes folder navigator
status: To Do
assignee: []
created_date: '2026-08-13 01:43'
labels:
  - notes
  - folders
  - ui
dependencies:
  - TASK-15702
references:
  - Docs/superpowers/specs/2026-08-12-notes-folder-import-sync-design.md
  - backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md
  - >-
    backlog/decisions/060-notes-sync-round-trip-and-interoperability-constraints.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose the local folder foundation in Library Database Notes as a lazy hierarchical tree with Unfiled, multiple note placements, breadcrumbs, stable focus, semantic status text, and visible manual folder actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Database Notes renders root folders, nested folders, notes, and Unfiled lazily with bounded bulk reads and no per-note service calls.
- [ ] #2 A note in multiple folders has distinct placement rows that open one note identity and show the correct breadcrumb.
- [ ] #3 Users can create, rename, move, remove, and restore folders; move notes; and add a note to another folder without deleting note content.
- [ ] #4 Managed and restored-without-owner placements are visibly distinguished, color-independent, and protected from manual mutation.
- [ ] #5 Selection, expansion, focus, and editor identity remain stable across refresh, reorder, resize, and compact 60x20 layouts.
- [ ] #6 Database work runs off the Textual event loop for file-backed databases while in-memory test databases keep their thread-local safety.
- [ ] #7 Rendered-frame, accessibility, host-routing, performance, and live restart tests pass without regressing the existing Database Note editor or File Notes.
<!-- AC:END -->

## Definition of Done

- [ ] Every acceptance criterion is checked with automated or recorded evidence.
- [ ] Focused tests, broader Library/Notes regressions, static analysis, and live TUI verification pass.
- [ ] Implementation Notes summarize the approach, files, trade-offs, and evidence.
- [ ] ADR-059 and ADR-060 are linked from the implementation plan and notes.
- [ ] A rendered-frame self-review confirms hierarchy and status remain legible at 60×20 without color.
- [ ] The task is set to Done only after all requirements above are complete.
