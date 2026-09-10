---
id: TASK-32225
title: >-
  Library below 64 columns: Escape on a list is a no-op and the '‹ Library'
  return is never advertised
status: Done
assignee: []
created_date: '2026-09-10 14:55'
updated_date: '2026-09-10 19:24'
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
the Library pane closed, and the footer still advertised "esc focus rail" --
a chip whose destination (`#library-search-input`) lives inside that closed
pane, so the key moved nothing on Media, Prompts, Skills, Collections and
Conversations alike. The only way back was Media's "‹ Library" control, which
no footer named and which the other four destinations do not have.

Escape now reopens the Library pane through that control's own seam (the
shell's `PaneToggleRequested`, so the preference write and its persistence
generation still happen exactly once) and the footer reads "esc back to
Library", matching the control's label. A new `library_narrow_stage_return`
binding declared above `library_blur_text_field` and `library_list_focus_rail`
takes the key by declaration order -- the idiom this file already uses for
`library_media_bulk_delete_cancel` -- and is bounded to the same below-64 band
task-32065 gave the control, so the pinned step-back through visible panes
("focus Items" / "focus Library") keeps the key at every ordinary width.
Nothing in `_library_list_focus_rail_target` or its two Escape call sites was
touched; those are the crit9-notes branch's carve-out.

Two defects the unit test could not see, both found live at 60x24:
* The chip was registered but never painted. `AppFooterStatus` keeps a PREFIX
  of the context actions and drops the tail once it cannot fit -- always, at
  this width -- so an appended chip lost to "/ focus search" and "F6 next
  pane", neither of which does anything with the rail pane closed. It is now
  first (recovery before navigation, which is the order that ladder assumes).
* The chip went stale both ways: it was decided during `compose_content`,
  before the shell had resolved its allocation, and a pane toggle changes only
  CHILD widths so the shell's `on_resize` (and `AdaptiveReaderShellResized`)
  never fires. `LibraryAdaptiveReaderShell.sync_layout` -- the single place an
  applied layout is installed -- now posts `LibraryPaneVisibilityChanged`, and
  the screen re-registers the footer on it.

Live evidence (scratchpad crit9/wave/shell/caps/32225-*): at 60x24 on Prompts
and Skills the footer reads "esc back to Library | F1 · F6 · Ctrl+P · Ctrl+Q",
Escape brings the rail back, and the chip is gone from the next frame.

Files: tldw_chatbook/UI/Screens/library_screen.py,
tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py,
tldw_chatbook/Widgets/Library/__init__.py,
Tests/UI/test_library_crit9_shell.py, Docs/User_Guide/library.md.
<!-- SECTION:NOTES:END -->
