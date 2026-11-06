---
id: TASK-38981
title: "Library reader: opening the list beside an open item leaves the work pane a sliver below 80 columns, and Prompts keeps an empty work pane below 64"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:42
updated_date: 2026-11-06 09:00
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
- **Shell.** `AdaptivePaneShell.sync_layout` hides the work pane whenever another pane holds the whole stage (`reader_width == 0` with a pane open), so none of its controls stay in the focus chain at zero width. The item stays mounted.
  - Focus inside the work pane moves into the pane that took the stage.
  - When the pane shows again, focus returns to the last work-pane focus (tracked in `on_descendant_focus`), or the first work-pane control, if focus was on a grip, was None, or was being evacuated.
  - A shell whose focus reveals a zero-width reader opts out with `hide_zero_width_work = False`. That is Artifacts' `LibraryArtifactsReaderShell`: F6 or Tab into its body runs `reveal_focused_reader`, as `test_narrow_f6_reveals_reader_and_returns_through_items_grip` pins. My first cut hid every zero-width work pane with no opt-out and turned those tests red. The second cut hid only under an explicit list priority, which left the empty narrow stage's controls as invisible Tab stops; the review found that.
- **Profiles.** The flag is on every Library profile but File Notes: Notes, Media, Prompts, Skills, Collections, Conversations and Artifacts. `list_first_when_empty` is added to Prompts (AC#4) and, as TASK-34851 asks, to Collections, Conversations and Skills. The stage applies while the whole stage fits the list's 72-cell ceiling: up to 82 columns, not only below 80.
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



### Review fixes (r10l-rev1)

- **CRITICAL: an item picked from the list holding the stage stayed hidden** (Media, Conversations, Collections). The layout kept its explicit `priority_pane="items"`, and later resolves inherited it.
  - A small `list_holds_the_stage(layout)` helper in `adaptive_reader_state` recognises that layout.
  - An explicit open drops the priority so the opened item takes the stage back:
    - Media: `_open_library_media_viewer` (a row's Enter or click). A focus move's preview select keeps the list, so moving through the rows with the arrow keys still browses it. My first cut released the stage in the open-time width restore, which also runs for the preview select. The list then came back through the grip.
    - Conversations: `_start_library_conversation_reader_selection`, through `_hand_the_list_stage_to_the_reader`.
    - Collections: `_select_library_collection_capture`.
  - Notes, Skills and Prompts already dropped it.
  - Test: `test_an_item_picked_from_the_stage_list_takes_the_stage_back[notes|media|conversations|collections|skills|prompts]` at 60 columns. It opens the list beside an open item, presses Enter on another row, and requires the new item open, the work pane shown at the profile floor or wider, and focus on a visible control. Red on HEAD for media, conversations and collections ("work shown False", the layout still priority "items"). Prompts was red only on my stricter first assertion that focus land inside the item: Prompts and Media leave it on a grip, and the test now asks for a visible control.
- **Important: invisible Tab stops in the empty narrow stage** (Skills, Conversations with none). The shell now hides the zero-width work pane in that stage too, with the opt-out above. Test: `test_an_empty_narrow_stage_leaves_no_invisible_work_tab_stop[collections|skills|prompts|conversations-empty]`. On HEAD it was red with `('skills', ['library-skill-work-pane'])` and `('conversations-empty', ['library-conversation-reader-read', '…-info', '…-find', '…-messages'])`.
- **Important: Artifacts.** The flag is on `ARTIFACT_READER_PROFILE`, and AC#1 now holds for Artifacts too, so it stays ticked. Its reader stays displayed at zero width while the list holds the stage, and F6 or Tab into the body reveals it at full width, as before. Evidence:
  - the reviewer's Artifacts suites (focus, canvas, navigation and navigation-mounted): 46 of 46;
  - `Tests/UI/test_library_artifacts_focus.py::test_items_opened_beside_an_open_artifact_take_the_stage[size0|size1|size2]` (80, 64 and 50 columns), red with the flag off (`items_width=32, reader_width=8` at 50);
  - the resolver pin `test_an_opened_list_beside_an_open_item_never_leaves_it_a_sliver`, which now covers seven profiles, Artifacts included, at every width 40-82.
- **Minor.** The User Guide and these notes say "up to 82 columns". The two stale comments are corrected (the Media resolver test's "narrow 32", and the CI pin's times). The pre-import record now credits the modules that are in the library route: `adaptive_pane_shell` +43, `screen_constants` +17, `library_screen` +3, `library_media_reader_state` +2, total 426,539 → 426,604. `adaptive_reader_state` is not in the route. The snapshot was regenerated with `--only preimport`.
- **Changed-files run** (the reviewer's 7 files plus the Artifacts, Conversations, Collections, Media and reader-shell suites; 25 files, `-n 4`, load 8-15): mine 324 failed, 1168 passed; `fu-base-r10` 334 failed, 1092 passed. Nothing fails only on mine. Ten fail only on base: the round's fixed reds (the modal inventory and the trust dialog) and one route-cycle flake.
- **New nodes** at the census tail: the six picks, the four empty-stage cases, the three Artifacts sizes and TASK-38983's Retry conflict test. Floor 1737.
- Files: `tldw_chatbook/Widgets/adaptive_pane_shell.py`, `tldw_chatbook/Utils/adaptive_reader_state.py`, `tldw_chatbook/UI/Library_Modules/library_media_controller.py`, `tldw_chatbook/UI/Library_Modules/library_conversation_reader_controller.py`, `tldw_chatbook/UI/Library_Modules/library_collections_controller.py`, `tldw_chatbook/UI/Library_Modules/library_artifacts_controller.py`, `tldw_chatbook/Widgets/Library/library_artifacts_reader_shell.py`, `Tests/UI/test_library_adaptive_reader_closeout.py`, `Tests/UI/test_library_artifacts_focus.py`, `Tests/Library/test_library_adaptive_reader_state.py`, `Tests/Library/test_library_media_reader_state.py`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/Performance/test_screen_preimport_payload_budget.py`, `Tests/Performance/boot_budget_snapshots/preimport_payload.json`, the census files, and four `Docs/User_Guide/library/` pages.
- **Follow-up.** After a pick from the stage list, Media leaves focus on its Items grip and Prompts on its Library grip, both visible. Focus could go into the opened item instead.

