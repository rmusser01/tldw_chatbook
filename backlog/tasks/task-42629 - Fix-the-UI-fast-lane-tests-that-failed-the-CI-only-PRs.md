---
id: TASK-42629
title: Fix the UI fast-lane tests that failed the CI-only PRs
status: Done
assignee:
  - '@claude'
created_date: '2026-11-27 18:30'
updated_date: '2026-11-27 18:30'
labels:
  - tests
  - flaky
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On 2026-11-26/27, the PR runs of #3777, #3788, #3789 and #3791 went red in UI Fast Lane shards 3, 9, 20, 36 and 37. Three of those PRs only changed CI files and could not reach the failing tests. The same files passed locally on dev. A full-tier PR that fails a shard twice is evicted from the merge queue after hours of CI, so these failures cost whole queue cycles, and the owner asked for them to be fixed.

The tests should fail only when the behaviour they pin is broken, not when a loaded runner reads the screen before a deferred refresh lands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The slash-command popup sits above the composer after its first show, even when a paint hold or a later reflow comes between the show and its layout
- [x] #2 The Send-strip tests wait for the deferred control-bar replay's painted strip instead of reading it after one pause
- [x] #3 The zero-match outcome-line test waits for the forced transcript rebuild's paint
- [x] #4 The folder-picker tests click only once the press handler's own precondition (a selected tree row) holds, and name the state if the picker still does not open
- [x] #5 The projection test names the sync's entry guards when its second full sync defers
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read each CI failure and the code it depends on.
2. Reproduce each one locally (probes, injected holds) before changing it.
3. Fix the cause where it was found; where it was not reproduced, wait on the handler's own precondition or add diagnostics, and say so.
4. Run every test file that touches the popup, on the branch and on dev, and the light job's targets.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Popup (shard 37, `test_popup_anchors_above_composer_with_real_css`, `assert -12 >= 0`). Product fix.** `ConsoleCommandPopup.reposition` worked out its flow slot as `region.y` minus the last offset it had applied. That is right only after a layout has applied that offset. A probe traced the first show: the first two repositions ran before any layout, and only the 0.1 s timer reposition saw a laid-out region. Under a held `batch_update` all six repositions, the timer's included, ran before layout, and the popup stayed in its flow slot under the composer (y=82 against composer y=45) until the next keystroke. CI's y=-12 is the same math caught by a reflow after the last reposition. The popup now sets `absolute_offset` (screen coordinates, no flow slot) and re-anchors from `Screen.screen_layout_refresh_signal`, as `console_message_more_menu.py` already does. The `call_after_refresh` and the 0.1 s timer are gone. Ceiling (a `ponytail:` comment): Textual reads `Offset(0, 0)` as unset, so a popup that would sit at the screen's very top-left corner falls back to its flow slot; that needs the composer within 8 rows of the top. `test_a_paint_hold_across_the_first_show_still_anchors_the_popup` holds a batch across the first show. It fails on dev's popup (y=82) and is added to `scripts/ui_pr_gate_census.txt`, since lanes run this file by node. The three anchor tests wait until the popup is laid out at its anchor, not 0.2 s.

**Send strip (shard 20, two tests in `test_console_send_strip_codes.py`).** The tests' direct `_sync_console_control_bar()` defers to the coalesced control-bar replay: a probe saw it return False in every run. The strip then paints when the replay lands, and one `pilot.pause()` read an empty strip whenever the replay landed later. Dev's version failed 1 run in 3 locally. `_synced_strip` waits until the strip's content and its painted text both read the expected copy. Waiting for the text alone was not enough: the strip passes through a narrower, cut paint, and the click test clicked before layout. All four sites that read the strip after a direct sync use it. 4 of 4 runs passed.

**Outcome line (shard 3, `test_a_zero_match_send_shows_its_line_and_keeps_it_after_a_reload`).** The forced `_sync_native_console_transcript(force=True)` returns False when live work defers it, and the test read the painted screen after one pause. It now waits with `until`, as the step before it already did. Holding paints for 0.3 s did not make dev's version fail locally, so the exact deferral on CI was not reproduced.

**Folder picker (shard 9, `test_move_note_picker_offers_nested_folders_on_a_fresh_visit_and_narrows_by_typing`).** The click on Move was lost: the picker did not open in 10 s. Both press handlers re-read the selected tree row and return without a word when it is gone, while the button's enabled state comes from the canvas build. `_open_picker` now also waits for a selected row, and the timeout names the top screen, the selected row and the painted screen. Not reproduced locally: 8 probes under `-n 4` showed one button instance, enabled at once, with the note selected throughout.

**Projection (shard 36, `test_a_full_sync_slower_than_the_display_life_does_not_replay_itself`).** The second full sync returned False on CI. In 12 local probe runs the entry guards were clear and it returned True. The assertion now names the guards (sync in progress, sync requested, replay, maintenance pause, projection pending), so the next failure says which path deferred.

**Evidence.**
- The 19 files that touch the popup: 156 failures on the branch and the same 156 on dev d2ad1e5330 (existing reds in unlaned files such as `test_console_rewind_restore.py`), none new.
- The light job's 51 targets for this change set: 185 passed.
- `test_console_command_popup.py`, `test_console_send_strip_codes.py`, `test_console_library_outcome_line.py`, `test_library_notes_folder_picker_lists_every_folder.py` and `test_console_projection_reads_off_loop.py` pass with `-n 4`.

Files: `tldw_chatbook/Widgets/Console/console_command_popup.py`, `Tests/UI/test_console_command_popup.py`, `Tests/UI/test_console_send_strip_codes.py`, `Tests/UI/test_console_library_outcome_line.py`, `Tests/UI/test_library_notes_folder_picker_lists_every_folder.py`, `Tests/UI/test_console_projection_reads_off_loop.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py` (floor 2811 -> 2812).
<!-- SECTION:NOTES:END -->
