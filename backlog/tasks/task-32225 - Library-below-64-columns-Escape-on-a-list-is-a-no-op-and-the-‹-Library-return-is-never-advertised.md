---
id: TASK-32225
title: >-
  Library below 64 columns: Escape on a list is a no-op and the '‹ Library'
  return is never advertised
status: Done
assignee: []
created_date: '2026-09-10 14:55'
updated_date: '2026-09-10 20:20'
labels:
  - library
  - layout
  - keyboard
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In the single-stage layout Escape on a list does nothing and the footer never names the return control. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 24.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Escape (or an advertised key) returns to the rail stage; the footer names it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test at 60x24: the footer advertises 'esc focus rail', the rail pane is closed, and Escape moves nothing.
2. Add the narrow single-stage context to _library_route_shortcuts_for_current_state ('esc back to Library') and route Escape to the same seam the '< Library' control uses.
3. Docs + live-verify at 60 columns.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Below 64 columns every adaptive-reader destination collapses to one stage with
the Library pane closed, and the footer advertised "esc focus rail" -- a chip
whose destination (`#library-search-input`) lives inside that closed pane, so
the key moved nothing. The only way back was Media's "‹ Library" control,
which no footer named and which the other destinations do not have.

Escape now reopens the Library pane through that control's own seam (the
shell's `PaneToggleRequested`, so the preference write and its persistence
generation still happen exactly once) and the footer reads "esc back to
Library". A new `library_narrow_stage_return` binding declared above
`library_blur_text_field` and `library_list_focus_rail` takes the key by
declaration order -- the idiom this file already uses for
`library_media_bulk_delete_cancel`. Nothing in `_library_list_focus_rail_target`
or its two Escape call sites was touched; those are the crit9-notes carve-out.

The gate is three conditions, and fix round 1 added the last two:
1. the below-64 band, the same floor task-32065 gave the "‹ Library" control,
   so the pinned step-back grammar keeps the key at ordinary widths;
2. a positive width -- `self.size` is `Size(0, 0)` until the screen is in the
   layout map and the width contract REFUSES that, and this gate runs from
   `compose_content` and from a message handler, both reachable before the
   first measure (review finding 2);
3. NO earlier Escape binding is live (review finding 1). Eleven are declared
   above this one, and Textual gives the key to the first gate that passes, so
   at 60x24 the Media viewer painted "esc back to Library" while Escape went
   to the media list. The gate now stands down the way the `emergency.enabled`
   branch beside it does, reading BINDINGS through the existing
   `_library_escape_actions_before` rather than a second hand-written roster.

Two delivery defects only live verification caught:
* the chip was registered but never painted -- `AppFooterStatus` keeps a
  PREFIX of the actions and drops the tail, always at this width. It is now
  first (recovery before navigation), which costs the canvas's other chips at
  60 columns; that trade is now documented in `library.md`.
* the chip went stale in both directions -- decided during `compose_content`
  before the shell resolved its allocation, and a pane toggle changes only
  CHILD widths so the shell's `on_resize` never fires.
  `LibraryAdaptiveReaderShell.sync_layout` now posts
  `LibraryPaneVisibilityChanged` and the screen re-registers the footer on it.

Live evidence, re-captured in fix round 1 (60x24, seeded profile):
* Prompts before/after Escape -- "esc back to Library | F1 · F6 · Ctrl+P ·
  Ctrl+Q", then the rail is back and the chip is gone
  (caps/32225-prompts-60-before-esc.txt / -after-esc.txt);
* the Media viewer with an item open shows the VIEWER's chips, not the return
  (caps/32225-media-viewer-60-p1.txt), and once Escape is genuinely inert
  there the return chip appears and does return.

Guide note: Notes keeps its own Escape at this width (its
`library_notes_escape` is one of the eleven), so `library.md` now says a
surface that owns Escape keeps it, rather than claiming all six destinations
return to the rail.

Files: UI/Screens/library_screen.py,
Widgets/Library/library_adaptive_reader_shell.py, Widgets/Library/__init__.py,
Tests/UI/test_library_crit9_shell.py, Docs/User_Guide/library.md.
<!-- SECTION:NOTES:END -->
