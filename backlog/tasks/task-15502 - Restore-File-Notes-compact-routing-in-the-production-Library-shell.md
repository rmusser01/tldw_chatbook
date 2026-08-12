---
id: TASK-15502
title: Restore File Notes compact routing in the production Library shell
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 20:56'
updated_date: '2026-08-11 21:23'
labels:
  - library
  - notes
  - filesystem
  - ux
  - responsive
  - regression
dependencies: []
references:
  - >-
    backlog/tasks/task-2850 -
    Notes-Files-mode-strands-the-user-outside-the-Library-frame.md
  - >-
    .impeccable/critique/2026-08-11T20-58-28Z__ok-widgets-library-library-file-notes-workspace-py.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A final merged-dev acceptance pass at 5bf47b2f found a regression at the supported 40x20 viewport. Notes Files mode remains inside the Library shell, but compact single-stage routing recognizes only Database Notes, so the 120-column rail and 40-column-minimum File Notes canvas are mounted side by side and the canvas starts offscreen. Restore keyboard-reachable File Notes without changing wide-layout behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 40x20, entering Notes Files mode presents one visible Library stage at a time and the File Notes canvas intersects the viewport.
- [x] #2 A keyboard-only user can enter File Notes, reach its first meaningful control, and return to the Library rail or Database Notes without pointer input.
- [x] #3 Resizing between 40x20, 120x40, and 160x45 preserves the active File Notes route, selection, editor state, and a valid visible focus target.
- [x] #4 Mounted tests exercise the production LibraryScreen shell and assert rendered viewport intersection and keyboard reachability at 40x20; component-only harnesses do not count as the compact acceptance proof.
- [x] #5 Wide and moderate Library layouts retain their intended rail-and-canvas presentation, and focused static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the 40x20 production-shell geometry and trace Files entry, compact-stage selection, resize, and return focus before editing.
2. Extend compact Notes-stage ownership to the Files source while preserving the mounted Library shell and wide rail-and-canvas layout.
3. Add CSS-backed production LibraryScreen tests for 40x20, 120x40, and 160x45 that cover keyboard entry, focus reachability, resize retention, and return navigation; confirm the compact assertions fail before the fix.
4. Run the focused responsive and adjacent Database Notes tests, Ruff, compileall, diff checks, and full-app production-shell acceptance.

ADR required: no

ADR path: N/A

Reason: This is a routine responsive-routing regression fix within ADR-011, ADR-015, and ADR-031; it changes no storage, ownership, service, or long-lived application boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored compact File Notes routing by treating Database and Files as equal owners of the Library Notes canvas stage. A post-recompose responsive measurement prevents the full application shell from retaining stale wide-layout state during compact startup. Below 120 columns the rail remains mounted but hidden while File Notes owns the visible canvas; at 120 columns and wider the rail-and-canvas presentation remains unchanged.

Added production-shell coverage for 40x20 entry, keyboard reachability and Escape return, plus 40x20/120x40/160x45 resize round trips that retain the open file, editor instance, text, and visible focus. The pre-fix run failed with the canvas beginning at x=120 and extending to x=640 in a 40-column viewport; the final focused matrix passed 14 tests. Ruff passed with the file's seven unrelated pre-existing E721 findings excluded, compileall passed, and `git diff --check` passed. Updated the Library and File Notes guides to describe the compact one-stage layout.

ADR required: no. ADR-011, ADR-015, and ADR-031 already govern the responsive shell, visual hierarchy, and keyboard behavior. No storage, service, ownership, or long-lived application boundary changed.

Modified files: `tldw_chatbook/UI/Screens/library_screen.py`, `Tests/UI/test_library_file_notes_workspace.py`, `Docs/User_Guide/library.md`, and `Docs/User_Guide/library/file-notes.md`.
<!-- SECTION:NOTES:END -->
