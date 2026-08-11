---
id: TASK-15502
title: Restore File Notes compact routing in the production Library shell
status: To Do
assignee: []
created_date: '2026-08-11 20:56'
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
- [ ] #1 At 40x20, entering Notes Files mode presents one visible Library stage at a time and the File Notes canvas intersects the viewport.
- [ ] #2 A keyboard-only user can enter File Notes, reach its first meaningful control, and return to the Library rail or Database Notes without pointer input.
- [ ] #3 Resizing between 40x20, 120x40, and 160x45 preserves the active File Notes route, selection, editor state, and a valid visible focus target.
- [ ] #4 Mounted tests exercise the production LibraryScreen shell and assert rendered viewport intersection and keyboard reachability at 40x20; component-only harnesses do not count as the compact acceptance proof.
- [ ] #5 Wide and moderate Library layouts retain their intended rail-and-canvas presentation, and focused static checks pass.
<!-- AC:END -->
