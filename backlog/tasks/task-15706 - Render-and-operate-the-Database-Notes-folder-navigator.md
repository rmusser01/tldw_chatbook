---
id: TASK-15706
title: Render and operate the Database Notes folder navigator
status: In Progress
assignee: []
created_date: '2026-08-13 01:43'
labels:
  - notes
  - folders
  - ui
dependencies:
  - TASK-15705
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

## Implementation Plan

ADR required: no
ADR paths: `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/060-notes-sync-round-trip-and-interoperability-constraints.md`
Reason: this task implements the folder UI, placement identity, ownership, and service boundaries already accepted by those ADRs; it does not introduce a new architectural choice.

1. Add a Textual-independent folder-tree projection that consumes bounded `NoteFolderPage` batches, preserves placement-aware identities, produces breadcrumbs and Unfiled rows, distinguishes managed/inactive-managed placements without relying on color, and reconciles expansion/selection/focus by stable identity.
2. Add focused unit tests for hierarchy, duplicate placements, generated-owner ancestor collapsing, manual duplicates, paging metadata, semantic labels, and stable state reconciliation; observe each test fail before implementing its behavior.
3. Extend `LibraryNotesCanvas` with a keyboard-operable lazy navigator plus compact folder action surfaces. Keep note buttons compatible with the existing editor/select handlers while carrying a separate placement identity and breadcrumb.
4. Add screen-owned async loading and mutation orchestration through `NotesScopeService`, including stale-result guards, bounded batch reads, file-backed off-loop behavior, thread-safe in-memory test handling, and actionable capability/error states.
5. Wire create, rename, move, remove, restore, move-note, and add-placement actions. Protect managed/inactive-managed placements from manual detachment or destructive folder actions, and refresh without losing surviving expansion, placement focus, note selection, or editor identity.
6. Add widget, screen, host-routing, accessibility, compact 60x20 rendered-frame, performance, and regression tests for Database Notes and File Notes. Correct the pre-existing multiselect fixture only if the touched handler requires it.
7. Run focused and broader Library/Notes suites, static checks, a real file-backed smoke path, and live TUI restart verification. Review the rendered 60x20 frame without color, document evidence and trade-offs, link ADR-059/060 in Implementation Notes, check every criterion, and only then set TASK-15706 to Done.
