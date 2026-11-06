---
id: TASK-38981
title: "Library reader opening the list beside an open item leaves the work pane a sliver below 80 columns and Prompts keeps an empty work pane below 64"
status: To Do
created_date: 2026-11-06 01:42
labels:
- follow-up
- library
- ux
- accessibility
priority: medium
references:
- tldw_chatbook/Utils/adaptive_reader_state.py
- tldw_chatbook/Widgets/adaptive_pane_shell.py
- tldw_chatbook/UI/Library_Modules/screen_constants.py
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_prompts_controller.py
- Tests/Library/test_library_adaptive_reader_state.py
dependencies:
- TASK-38757
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down. Checked on dev 29592ecc9b and on the TASK-38757 branch, 2026-11-05. TASK-38757 fixes the `width - 42` reader-pane sliver only inside the File Notes workspace. The shared resolver and shell it leaves alone serve every other Library destination: `tldw_chatbook/Utils/adaptive_reader_state.py`, `tldw_chatbook/Widgets/adaptive_pane_shell.py`, `library_screen.py` and `UI/Library_Modules`.

**With no item open, the list views are fine.**
- The Notes list (`library_screen.py:6915-6926`) and Media Trash (`:7678-7693`) pass `priority="items"` with `reader_has_item=False`.
- Their profiles (`LIBRARY_NOTES_READER_PROFILE`, `MEDIA_READER_LAYOUT_PROFILE`, `tldw_chatbook/UI/Library_Modules/screen_constants.py:59-67`) set `list_first_when_empty`, so below 64 columns the list takes the stage (`adaptive_reader_state.py:331-357`, TASK-32389).
- Measured on the real Library screen for the Notes list: the work pane is 1 cell at 40 and 60, then 22 at 64 and 37 at 79. It has no focusable widgets.
- Media Trash was checked through the resolver only and gives the same split.

**With an item open, opening the list leaves a sliver.**
- With `reader_has_item=True`, every one of the 7 profiles resolves to a 32-cell list and a `width - 42` work pane.
- Notes, with a note open and the list grip pressed (load 16.8):

  | Width | Work pane | Work-pane controls outside the visible area |
  |---|---|---|
  | 40 | 1 cell | 6 of 9 (back, edit, preview, context, save, use-in-console) |
  | 50 | 8 | 5 |
  | 60 | 18 | 1 |
  | 70 | 28 | 0 |

- Media, with the viewer open and the list grip pressed (load 7.1):

  | Width | Work pane | Work-pane controls outside the visible area |
  |---|---|---|
  | 40 | 1 cell | 6 of 12 |
  | 60 | 18 | 3 |
  | 79 | 37 | 1 |

- This is the grip-from-editor case that TASK-38757 rev3 fixed for File Notes alone. A few of the counted controls sit in horizontally scrolling strips, so the widths are the firm evidence.

**The Prompts list keeps an empty work pane.**
- `library_prompts_controller.py:800` sets `priority="items"` in list view, but `LIBRARY_PROMPTS_READER_PROFILE` has no `list_first_when_empty`.
- Measured: an empty 8-cell work pane at 50 columns, 18 at 60 and 22 at 64, beside the 32-cell list.
- This is the TASK-32389 case that Notes and Media already handle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With an item open in Notes, Media or any other shared-shell destination, opening the list at every width from 40 to 79 either gives the list the stage (work pane hidden, item kept) or leaves the work pane at least the destination's usable minimum; red before: the work-pane widths recorded above
- [ ] #2 While the list is open at those widths, Tab never lands on a work-pane control outside the visible region
- [ ] #3 Closing the list (or Back) returns to the open item with focus inside it
- [ ] #4 The Prompts list below 64 columns gives the list the stage as Notes and Media do, and opening a prompt hands the width back
- [ ] #5 A 40-79 column sweep on the real Library screen for Notes and Media pins #1-#4 and is red on dev
<!-- AC:END -->
