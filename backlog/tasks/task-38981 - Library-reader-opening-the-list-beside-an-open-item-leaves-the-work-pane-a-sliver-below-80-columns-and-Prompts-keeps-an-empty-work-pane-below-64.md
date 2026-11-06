---
id: TASK-38981
title: "Library reader: opening the list beside an open item leaves the work pane a sliver below 80 columns, and Prompts keeps an empty work pane below 64"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:42
updated_date: 2026-11-06 07:04
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
- This is the TASK-32389 case that Notes and Media already handle. It needs the same `list_first_when_empty` fix as open TASK-34851 (a duplicate of TASK-32304), which covers Collections, Conversations and Skills. One PR can do Prompts together with them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With an item open in Notes, Media or any other shared-shell destination, opening the list at every width from 40 to 79 either gives the list the stage (work pane hidden, item kept) or leaves the work pane at least the destination's usable minimum; red before: the work-pane widths recorded above
- [x] #2 While the list is open at those widths, Tab never lands on a work-pane control outside the visible region
- [x] #3 Closing the list (or Back) returns to the open item with focus inside it
- [x] #4 The Prompts list below 64 columns gives the list the stage as Notes and Media do, and opening a prompt hands the width back (the same fix as TASK-34851 applies to Collections, Conversations and Skills)
- [x] #5 A 40-79 column sweep on the real Library screen for Notes and Media pins #1-#4 and is red on dev
<!-- AC:END -->

## Implementation Plan

1. Tests first, on the real Library screen through the closeout harness (`@private_profile_test`): for Notes and Media, open an item, then at every width 40-79 open the list. Check that the work pane is hidden or at the profile floor, that no work-pane control sits outside the visible region, and that closing the list returns focus inside the item. For Prompts, check that the list takes the narrow stage and an opened prompt takes it back. Show them red on base.
2. Resolver: an opt-in profile flag, `list_stage_over_starved_work`. With it, an explicit list request with an item open that would leave the work pane under `work_min_width` gives the list the stage, while that fits `list_max_width`. Default off, so File Notes (TASK-38757's own rule) and the golden grid's literal profiles are unchanged.
3. Shell: hide the work pane when a pane holds the whole stage (`reader_width == 0`), move focus out of it, and return focus to the item when it shows again.
4. Profiles: the flag on every Library destination except File Notes. `list_first_when_empty` on Prompts (AC#4), and on Collections, Conversations and Skills, the same fix TASK-34851 asks for.
5. Resolver unit pins for every profile at 40-79, the shell pin, the User Guide, the census tail, and covering runs.

## Implementation Notes

- **Resolver.** `AdaptiveReaderLayoutProfile` gains an opt-in `list_stage_over_starved_work`. Take an explicit list request (`priority="items"`) with an item open, in the width-starved branch. If it would leave the work pane under the profile's `work_min_width`, and the whole stage fits `list_max_width`, the list takes the stage and the work pane gets 0 columns. The default is off. File Notes resolves its narrow navigator with the same priority and keeps TASK-38757's rule: 18 cells beside the navigator from 60 columns. The golden-grid digest is unchanged, because its literal profiles do not set the flag.
- **Shell.** `AdaptivePaneShell.sync_layout` hides the work pane when an explicit list priority holds the whole stage (`reader_width == 0`, items open, `priority_pane == "items"`), so none of its controls stay in the focus chain at zero width. The item stays mounted.
  - Focus inside the work pane moves into the list.
  - When the pane shows again, focus returns to the last work-pane focus (tracked in `on_descendant_focus`), or the first work-pane control, if focus was on a grip, was None, or was being evacuated.
  - It is scoped to the explicit priority. An empty list's narrow stage (`list_first_when_empty`, no priority) keeps its zero-width work pane, because Artifacts' F6 reveals its reader through it (`test_narrow_f6_reveals_reader_and_returns_through_items_grip`). My first cut hid every zero-width work pane and turned those Artifacts tests red.
- **Profiles.** The flag is on Notes, Media, Prompts, Skills, Collections and Conversations: the six Library profiles beside File Notes in the task's "7 profiles". `list_first_when_empty` is added to Prompts (AC#4) and, as TASK-34851 asks, to Collections, Conversations and Skills. Artifacts is not changed; see the follow-ups.
- **Tests.**
  - AC#5, on the real Library screen through the closeout harness (`@private_profile_test`): `test_opening_the_list_beside_an_open_item_leaves_no_work_sliver[notes|media × 40|50|60|70]` covers every width from 40 to 79, ten per case. At each width it focuses a control in the open item, clicks the list grip, and checks three things: the work pane is hidden or at least the profile floor, no work-pane control in the focus chain is outside its clip, and focus is visible. It then clicks the grip again and checks that focus is inside the open item.
  - `test_every_other_list_destination_gives_an_opened_list_the_stage[prompts|skills|collections|conversations]` runs the same check at 60 and 44.
  - `test_prompts_list_takes_the_narrow_stage_and_an_open_prompt_takes_it_back` checks the Prompts list at 63-40 and an opened prompt at 40 (AC#4).
  - `test_an_empty_list_destination_keeps_the_list_at_60x24[collections|skills]` covers TASK-34851. Conversations opens its first conversation on entry, so it has no empty state there.
  - Unit pins: `test_an_opened_list_beside_an_open_item_never_leaves_it_a_sliver` (six profiles, 40-79), `test_file_notes_keeps_its_own_narrow_navigator_split`, `test_an_empty_narrow_stage_goes_to_the_list_for_every_list_destination`, and the shell's `test_a_pane_given_the_whole_stage_hides_the_work_pane_and_returns_to_it`.
- **Red on the base source** (the four source files at HEAD, with the tests).
  - Every Library-screen case fails with the recorded widths: `(40, 0)`, `(50, 8)`, `(60, 18)`, `(70, 28)` for Notes and Media; Prompts at 63 is `items_width=32, reader_width=21`; Collections and Skills at 60x24 are `items_open=False, reader_width=50`.
  - The unit pins fail too: 14 of 15. The 15th, the File Notes opt-out pin, guards an invariant dev already keeps. It went red (`(50, 0) == (32, 18)`) with the flag's default temporarily set to True.
- **Changed pins.** Each encoded the sliver beside an open item.
  - `test_an_items_priority_is_untouched_once_the_work_pane_has_something`: the list now takes the stage, and the priority is still kept.
  - Four in `Tests/Library/test_library_media_reader_state.py` (explicit open at 70, 60, 80 and 81): new exact widths, and the 80-to-160 return is pinned as "not the intermediate width".
- **Covering run** (36 files: the Library reader, shell, Notes, Media, Prompts, Skills, Collections, Conversations, Artifacts and File Notes suites, `-n 4`, load 25-45): 1652 failed, 1589 passed. Base: 1654 failed, 1552 passed. The failure sets match except for one load flake on mine, `test_loading_banner_paints_in_place_without_body_rebuild`, which passes 3 of 3 alone on mine and on base; base has three Media flow flakes of its own. Almost all the failures are the local `RecoveryRequired` trap.
- **Green times.** The sweep cases take 16-18 s each, and the other-destination and empty-list cases 9-30 s (`-n 4`, load 13-26). The fifteen nodes are appended at the census tail, floor 1722.
- **User Guide.** Media and Conversations, Notes, Prompts, Skills and Collections describe the list taking the stage beside an open item, and the narrow empty list.
- Files: `tldw_chatbook/Utils/adaptive_reader_state.py`, `tldw_chatbook/Widgets/adaptive_pane_shell.py`, `tldw_chatbook/UI/Library_Modules/screen_constants.py`, `tldw_chatbook/Library/library_media_reader_state.py`, `Tests/Library/test_library_adaptive_reader_state.py`, `Tests/Library/test_library_media_reader_state.py`, `Tests/UI/test_adaptive_pane_shell.py`, `Tests/UI/test_library_adaptive_reader_closeout.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, five `Docs/User_Guide/library/` pages, `backlog/docs/lessons-textual.md`.

