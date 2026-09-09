---
id: TASK-32127
title: >-
  Library Notes list pane is starved to about 38 columns at 235 wide while the
  work area holds about 145 empty columns
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 07:14'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - layout
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed by both assessors and the parent at 235x52: rail 34 columns, list 38, work area about 145 holding 'Select a note to edit it here.' This one geometry decision causes clipped titles ('Very long note — scaling laws d…'), the missing age column, the toolbar wrapping to three rows, hidden Move/Remove/Last import controls, the Sort strip losing its Title option and the clipped delete receipt (task-32123). 100x30 reads better than 235x52. Cause INFERRED (stylesheet not traced). Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With no note open, the list uses the canvas width and shows title, age and folder without clipping at 235 wide
- [x] #2 With a note open, the list keeps at least 40 percent of the canvas or 60 columns, whichever is smaller
- [x] #3 The toolbar fits on two rows at 235 wide with every folder and note action visible
- [x] #4 Geometry pinned by tests at 235x52 and 100x30
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin the resolver at 235 and 100 (no note open / note open) as failing tests.\n2. Give the Notes profile list_grows + a 64-cell comfort ceiling and pass reader_has_item from the notes view.\n3. Put the browse and transfer toolbars on one row so the toolbar is 2 rows.\n4. Live-verify at 235x52.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cause traced (the task had it as inferred): Notes resolved its layout with the default `reader_has_item=True` and a profile without `list_grows`, so at width 235 the resolver returned items_width=40 against reader_width=151 -- task-31979's "hand the empty Reader's width to the list" and task-31633's growth split were both already there, and Notes simply opted into neither.

Three changes: LIBRARY_NOTES_READER_PROFILE gets `list_grows=True` and `list_comfort_width=64`; `_sync_library_notes_reader_layout_from_shell` passes `reader_has_item=self._notes_state.view != "list"`; and the browse and transfer toolbars share one row. The Notes canvas also drops Button's 16-cell minimum, which by itself made "New / Sort: Newest / Select" 48 cells -- wider than the pane it sat in.

The merge is WIDTH-CONDITIONAL (review round 1): it applies from `_TOOLBAR_MERGE_MIN_WIDTH` (100 columns) up, measured from the resolved Items width the controller now passes to the canvas as `pane_width`. The first version gated on `compact` instead, which left the merged row clipping "Last import" off a 62-column pane -- exactly the width AC#2 guarantees with a note open -- and clipping the transfer actions off a 38-column one. Below the threshold the two groups keep their own rows, which is also why the shell's `test_library_shell_notes_list_actions_use_two_named_horizontal_rows` contract still holds at its 170-column harness (a ~78-column pane).

Pinned in Tests/UI/test_library_notes_wave_list.py: the resolver at 235 and 100 with and without a note open; the mounted toolbar at 137 columns (<= 2 rows, Move / Remove / Last import composed) and at 62 columns; and both frames assert over EVERY `.library-canvas-action` region in the frame rather than a hand-listed few -- the hand-listed version passed while two siblings were off-pane.

Live at 235x52 on the power profile: list pane ~137 columns with no note open (caps/01), ~62 with one open (caps/02), toolbar two rows, no title clipped. At 100x30 the compact stage keeps three rows with every action visible (caps/06).

The pinned decision test `test_only_the_media_profile_opts_into_list_growth` is reconciled: it now takes a named set of growth profiles rather than asserting Media is alone.

Files: tldw_chatbook/UI/Library_Modules/screen_constants.py, tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/UI/Library_Modules/library_notes_controller.py, tldw_chatbook/Widgets/Library/library_notes_canvas.py, tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated screen_agentic_library.tcss), Tests/Library/test_library_adaptive_reader_state.py, Tests/UI/test_library_shell.py, Tests/UI/test_library_notes_wave_list.py.
<!-- SECTION:NOTES:END -->
