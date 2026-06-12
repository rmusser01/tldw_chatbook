---
id: TASK-93
title: 'Rebuild Notes screen on destination workbench (Notes mode, feature parity)'
status: Done
assignee: []
created_date: '2026-06-10 03:16'
labels: []
dependencies:
  - TASK-82
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implements the Notes-mode workbench from Docs/superpowers/specs/2026-06-09-notes-workbench-design.md. ADR: none required.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Notes screen renders header + Notes/Sync/Templates mode strip + three-pane workbench
- [x] #2 Feature parity: search, keyword filter, sort, create blank/from template, import, edit title/content/keywords, auto-save, export/copy, delete, emoji, Use in Console
- [x] #3 Previously dead buttons (template-create, import, sort-order) work
- [x] #4 Mode switches preserve unsaved editor content
- [x] #5 Legacy Notes_Window untouched and functional
- [x] #6 Notes UI test suites pass
<!-- AC:END -->

## Implementation Plan

1. Extract list-population into NotesListPopulateMixin shared with legacy sidebar
2. New notes_workbench_panes.py: NotesNavigatorPane / NotesEditorPane / NotesInspectorPane
3. Rewrite NotesScreen.compose_content: ds header + Notes/Sync/Templates mode strip + display-toggled regions
4. Retarget sidebar-routed queries; add handlers for template-create, import, sort-order, Save All Changes, mode chips
5. Add active_mode to NotesScreenState with save/restore; TCSS additions + rebuild
6. Update pinned tests; add Tests/UI/test_notes_workbench_layout.py

## Implementation Notes

- Notes screen now renders the destination-workbench pattern: `.ds-destination-header` + purpose + slim status row (`{scope} | Notes`) + one-row Notes/Sync/Templates mode strip, then three panes (navigator 2fr | editor 5fr | inspector 2fr).
- Mode switching toggles pre-composed regions via `display` (no remount), so unsaved editor text survives mode flips (covered by test).
- Sync and Templates modes are honest stubs in this PR: Sync offers the existing sync dialog via an "Open Sync Tool" button; Templates points at the navigator's create-from-template. Full panes land with TASK-84.
- Fixed latent dead controls: create-from-template, import (FileOpen picker + `_parse_note_from_file_content`), sort-order toggle, and "Save All Changes" now have screen-level handlers (previously only the legacy `NOTES_BUTTON_HANDLERS` path served them, which the screen never dispatched).
- Inspector gains a note metadata block (created/modified/version/file-sync) updated on hydrate/clear.
- Legacy `UI/Notes_Window.py` untouched; `NotesSidebarLeft` now inherits the shared `NotesListPopulateMixin` (behavior-neutral); `NotesSidebarRight` untouched.
- Dropped legacy-only Load/Edit Selected buttons (ListView selection already loads) and the sidebar collapse toggles; `left/right_sidebar_collapsed` state fields kept for save/restore compatibility.
- Tests: 67 existing notes tests pass (3 updated for removed sidebar IDs), 9 new workbench layout/behavior tests, no regressions in destination/master-shell/focus suites (19 pre-existing unrelated failures unchanged).
- QA captures: `Docs/superpowers/qa/notes-workbench/`.
- ADR: none required — implements the design-system contract per Docs/superpowers/specs/2026-06-09-notes-workbench-design.md.
