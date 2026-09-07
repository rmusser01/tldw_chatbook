---
id: TASK-15774
title: Library media viewer search status and nav controls sit below the fold at 80x24
status: Done
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

- [x] At 80x24, the media viewer's search status and Prev/Next controls are
      reachable without requiring the user to already know to scroll or
      focus first (visible on open, or a clear, discoverable affordance)
- [x] The fix is a genuine layout decision (e.g. compact-mode chrome,
      collapsible content region, or reordered stack) rather than papering
      over the geometry with a scroll-into-view hack
- [x] `Tests/UI/test_library_shell.py`'s 170x48 and 80x24 in-place-navigation
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

## Implementation Notes

An active content search now DOCKS the whole search-controls block (box +
match count + ◀ Prev / Next ▶) to the top of the scrolling media viewer —
the find-bar convention, and the same pinned-edge chrome idiom the Library
ingest commit bar already uses (`#library-ingest-commit-bar`, `dock:
bottom`). An inactive search keeps the exact in-flow layout it has today,
so no space is reserved when nobody is searching and nothing about the
default viewer changes.

Why dock rather than compact/reorder: at 80x24 the stack above the
controls (Back, title, metadata, section header) exceeds the viewport by
itself, so any in-flow position can still be pushed out by content; and
the actual complaint was "can't see the match count or Prev/Next WHILE
USING them", which dock-on-active answers at every size and every scroll
position, not just on open. It is a layout decision, not a scroll hack —
no scroll_into_view anywhere.

Core changes:

- `tldw_chatbook/Widgets/Library/library_media_content.py` —
  `LibraryMediaContentSearchControls` toggles
  `-library-media-search-active` on ITSELF (constructor + both ends of
  `sync_query_state`), so the class rides the persistent container and
  survives the activity-flip child recompose.
- `tldw_chatbook/css/components/_agentic_terminal.tcss` (bundle
  regenerated via `build_css`, `check_bundle_sync` green) — the active
  class gets `dock: top`, raised background + bottom rule (scrolled
  content cannot bleed through the pinned bar), and collapsed child
  margins so the bar spends 6 rows of a 24-row terminal instead of 8.
- `Tests/UI/test_library_shell.py` — two born-red tests:
  `..._search_chrome_paints_at_compact_size` (compositor strips at
  exactly 80x24: "Match 1 of 101 matches" + both nav buttons painted,
  non-overlap with the content region, and still painted after pressing
  Next — red at HEAD `11646bba0` on `assert "◀ Prev" in painted`);
  `..._search_chrome_undocks_when_inactive` (120x40: inactive keeps Back
  above the search box in flow, active docks above the stack with exactly
  one search box painted, clearing the query releases the dock — red at
  HEAD on the dock assertion). Task-15458's 170x48 chrome test and 80x24
  in-place-navigation test stay green (its now-stale "below the fold
  until focused" comment updated to point here).
- `Docs/User_Guide/library/media-and-conversations.md` — pinned-bar
  behavior documented + Verified-against stamp (task/15774-burn @
  76e2b6c7e).

Evidence: before-strips at 80x24 show the search input at rows 17-19,
status barely at row 21 (only because focus had scrolled the input into
view), nav buttons at y=23 under the docked footer (not painted), content
body at y=24 — fully below the fold. After-strips show the docked bar at
rows 4-9 (input/count/nav + rule) with the flow (Back, title, metadata,
Content) below it; an extra probe (run, then deleted) showed the bar
still painted with the viewer scrolled to max (scroll_y=39/39), no
bleed-through. Full `Tests/UI/test_library_shell.py` serial run on this
branch: 591 passed / 3 failed, all three in the NOTES domain and none
mine — `test_library_note_keyboard_capability_matrix[create_discard-*]`
(x2) fails standalone at the base sha `11646bba0` AND at origin/dev tip
`31b0ef6a1` (pre-existing dev red, focus tab-walk stranded on `nav-lab`);
`test_library_shell_blank_note_autosaved_then_emptied_still_gcs_on_back`
passes standalone and passes in a clean 61-test file-order neighborhood
rerun (61/61) — its one serial failure coincided with concurrent pytest
probes loading the machine. The 34-test media viewer/search family passes
(includes task-15458's 170x48 chrome + 80x24 in-place-navigation tests),
`Tests/Library/test_library_media_content.py` 9/9, `check_bundle_sync`
green, ruff check clean on touched Python files (ruff-format diffs in
both files are pre-existing hunks identical at HEAD, left untouched to
keep the branch noise-free).
