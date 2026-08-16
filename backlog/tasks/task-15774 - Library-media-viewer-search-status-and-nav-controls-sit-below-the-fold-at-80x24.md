---
id: TASK-15774
title: Library media viewer search status and nav controls sit below the fold at 80x24
status: In Progress
assignee:
  - '@claude'
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

## Implementation Plan

1. Reproduce at exactly 80x24 with compositor render strips (real mounted
   `LibraryHarness` app, search active): capture the before state showing
   status + Prev/Next below the fold.
2. Add the born-red compositor test to `Tests/UI/test_library_shell.py`
   (80x24, search active, "Match 1 of 101 matches" + "◀ Prev" + "Next ▶"
   painted on-screen) mirroring task-15458's 170x48 chrome test; run at
   HEAD and record the red.
3. Layout decision: dock the search controls to the top of the scrolling
   viewer WHILE A SEARCH IS ACTIVE (browser/editor find-bar convention,
   and the same chrome idiom as the Library ingest commit bar's
   `dock: bottom` pinned edge). `LibraryMediaContentSearchControls`
   toggles an active-state class; the component CSS
   (`css/components/_agentic_terminal.tcss`, bundle regenerated via
   `build_css.py`, never hand-edited) docks the active controls with a
   raised background + separator rule and compact margins. Inactive
   search keeps today's in-flow layout: zero stolen space, no
   duplicated chrome.
4. Verify: new test green; task-15458's 170x48 chrome test and 80x24
   in-place-navigation test green; whole media-viewer family in
   `test_library_shell.py` green; 120x40 spot-check strips (inactive: no
   dock, no duplicate chrome; active: single docked bar).
5. ruff check + format touched files; update the Library User Guide
   page's Verified-against stamp if it documents this viewer; task notes
   + Done.
