---
id: TASK-32233
title: >-
  Library Notes: Escape is inert in the filter box and on the plain list
  although the footer promises 'esc focus rail'
status: Done
assignee: []
created_date: '2026-09-10 14:51'
updated_date: '2026-09-11 01:59'
labels:
  - library
  - notes
  - keyboard
  - ux
  - critique-9
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On the Notes canvas Escape in the filter box does nothing and the next printable key is typed into it (both profiles); on the plain Notes list Escape never reaches the rail's Search Library… box and typed keys are swallowed until Enter reopens the note. The critique-8 keyboard fix (task-32051) pinned Media only (`test_library_crit8_keyboard.py::test_escape_from_a_list_filter_box_still_goes_where_the_footer_says`); `_library_list_focus_rail_target()` returns the rail box for Notes but the hop does not take effect there. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Escape in the Notes filter box blurs to the canvas and the next printable key does its canvas job
- [x] #2 Escape on the plain Notes list (database and folder-tree layouts) focuses the rail's Search Library… box exactly as the footer says
- [x] #3 Both cases are pinned in `Tests/UI/test_library_crit8_keyboard.py` alongside the Media pins
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify live at the current dev tip (both cases, both layouts)
2. Pin both cases in Tests/UI/test_library_crit8_keyboard.py beside the Media pins (TDD red)
3. Fix the Escape ladder so the Notes list reaches the shared focus-rail hop
4. Live-verify at 235x52 and 100x30
5. Docs stamp + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Re-verified live at dev 1077ac2dad before changing anything: BOTH cases still reproduced. In the filter box Escape cleared the value and kept the caret, so `/ read esc n` put "n" straight back in; on the plain list `esc abc` went nowhere and the keys were swallowed. Captures under crit9/wave/notes/caps/32233-*.

Root cause was NOT the blur gate the snapshot suspected. `library_notes_escape` is the FIRST of this screen's many Escape bindings and its check_action is `visible_notes` -- true across the whole Notes workflow -- so Textual never tried `library_blur_text_field` or `library_list_focus_rail` (declared last) on this canvas at all. The action's own tail then restored the rail ROW through the notes focus identity, which resolved back into the navigator region and put the caret back in the filter.

Fix, in `action_library_notes_escape`'s tail only: when the canvas is genuinely showing its plain list, converge on the shared hop `/`, F6 and every sibling list canvas use (`action_library_list_focus_rail` -> `_library_list_focus_rail_target()`), and blur to the canvas through `action_library_blur_text_field` when the caret is already standing on that hop's destination. No Notes-specific Escape handler was added; the compact stage collapse above it is untouched (test_library_shell.py:21679 depends on it, and it was re-verified live at 100x30).

One delivery detail cost a round: the hop must be the canvas sync's explicit `then`, not `call_after_refresh`. A recompose already in flight restores the identity it captured before the key, and a bare deferred hop lands BEFORE that restore and is silently undone (traced through canvas_sync._restore_then_explicit).

AC#1 reading: Escape leaves the filter box for the rail's Search Library… box rather than straight to the canvas. That is the pinned house contract -- `test_escape_from_a_list_filter_box_still_goes_where_the_footer_says` (task-32051 fix round 1) forbids the blur binding outranking a hop that genuinely moves focus, and the wave plan's Step 3 named that same rule. A second Escape then blurs to the canvas and the next printable key does its canvas job, which is the AC's substance and is pinned too.

Files: tldw_chatbook/UI/Library_Modules/library_notes_controller.py, Tests/UI/test_library_crit8_keyboard.py (3 new pins beside the Media ones: filter box, plain list x2 layouts, rail box).
<!-- SECTION:NOTES:END -->
