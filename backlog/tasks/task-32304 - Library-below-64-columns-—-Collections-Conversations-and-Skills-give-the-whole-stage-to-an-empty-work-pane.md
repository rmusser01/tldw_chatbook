---
id: TASK-32304
title: >-
  Library below 64 columns — Collections, Conversations and Skills give the
  whole stage to an empty work pane
status: Done
assignee: []
created_date: '2026-09-11 00:54'
updated_date: '2026-11-06 13:11'
labels:
  - library
  - ux
  - critique-9
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Media fixed this for itself with `list_first_when_empty` (task-32065): with nothing open the narrow stage shows the list, not an empty reader. Collections, Conversations and Skills still hand the entire stage to their empty work pane at 60x24, so the user sees a blank canvas with the list hidden behind a hop. Observed while live-verifying task-32225 (critique-9 shell branch, PR #2585).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 60x24 with nothing open, Collections, Conversations and Skills show their list, matching Media's rule
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed by TASK-38981 (the Library list-stage work, round 10 of the 2026-10-23 burn-down), verified 2026-11-06. TASK-34851 and TASK-32304 are the same filing: identical text and the same created date. This one, the lower id, is the original; TASK-34851 is closed as its duplicate with the same evidence.

- **The rule.** TASK-38981 adds Media's `list_first_when_empty` to the Collections, Conversations and Skills reader profiles (`tldw_chatbook/UI/Library_Modules/screen_constants.py`), so with nothing open the narrow stage goes to the list.
- **The hidden work pane.** In that stage, `AdaptivePaneShell` hides the zero-width work pane, so its controls are not left in the focus chain.
- **Picking an item.** A pick from the stage list hands the stage back to the opened item, through the `list_holds_the_stage` check. Focus returns to the work pane only when the list closes.
- **AC#1 evidence**, all private-profile Library-screen tests in `Tests/UI/test_library_adaptive_reader_closeout.py`. Each is a UI census tail entry, recorded red on the pre-fix source in TASK-38981's notes.
  - Collections and Skills: `test_an_empty_list_destination_keeps_the_list_at_60x24[collections|skills]`. At 60x24 with nothing open, the list is open and the full stage width. Red before: `items_open=False, reader_width=50`.
  - Conversations, with no conversations: `test_an_empty_narrow_stage_leaves_no_invisible_work_tab_stop[conversations-empty]`. At 60x24 the list takes the narrow stage, and no work-pane control is a zero-width Tab stop. Red before: `('conversations-empty', ['library-conversation-reader-read', …])`.
    - With conversations, the Library opens the first one on entry, so there is no empty state there.
    - The same test covers `[collections|skills|prompts]`.
    - This node asserts the `items_open` layout flag; the resolver and shell pins below cover the list's display and width.
  - Resolver pin: `Tests/Library/test_library_adaptive_reader_state.py::test_an_empty_narrow_stage_goes_to_the_list_for_every_list_destination`.
  - Shell pin: `Tests/UI/test_adaptive_pane_shell.py::test_a_pane_given_the_whole_stage_hides_the_work_pane_and_returns_to_it`.
  - Picks after the list takes the stage: `test_an_item_picked_from_the_stage_list_takes_the_stage_back[notes|media|conversations|collections|skills|prompts]`.
- **Run.** These 17 nodes passed on the Library branch's tip through run.sh (`-n 4`, 39.8 s, load 6-9).
<!-- SECTION:NOTES:END -->
