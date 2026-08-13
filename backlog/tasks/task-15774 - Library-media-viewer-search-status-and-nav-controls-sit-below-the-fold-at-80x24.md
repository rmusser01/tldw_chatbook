---
id: TASK-15774
title: Library media viewer search status and nav controls sit below the fold at 80x24
status: To Do
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - ux
  - library
priority: low
---

## Description

Found during task-15458's 2026-08-13 macOS re-verification (input-latency
burn-down), recorded rather than fixed since it needs a layout decision: at
an 80x24 terminal, the media viewer's search status line and Prev/Next match
controls sit below the visible fold. Task-15458 already fixed the equivalent
issue at 170x48 (`ae757d8d4`, sizing the search-controls container to its
content), but that fix does not carry through to the compact 80x24 case —
the controls container is `height: auto` and the stack order is unchanged,
so at this size the content body still pushes the controls out of view. This
is the viewer's overall vertical density at small sizes, not a defect in
task-15458's in-place-navigation conversion itself (which task-15458's own
`..._inplace_navigation_holds_at_compact_size` test confirms: identity,
focus, and zero reparse all hold at 80x24 — the controls are simply
unreachable without scrolling/focusing first).

## Acceptance Criteria

- [ ] At 80x24, the media viewer's search status and Prev/Next controls are
      reachable without requiring the user to already know to scroll or
      focus first (visible on open, or a clear, discoverable affordance)
- [ ] The fix is a genuine layout decision (e.g. compact-mode chrome,
      collapsible content region, or reordered stack) rather than papering
      over the geometry with a scroll-into-view hack
- [ ] `Tests/UI/test_library_shell.py`'s 170x48 and 80x24 in-place-navigation
      tests (task-15458's) stay green; a new compositor-based test pins the
      80x24 chrome as visibly painted, mirroring the 170x48
      non-overlapping-regions test task-15458 already has
