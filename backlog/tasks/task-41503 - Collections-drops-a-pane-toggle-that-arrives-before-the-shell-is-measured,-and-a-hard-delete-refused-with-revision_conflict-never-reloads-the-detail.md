---
id: TASK-41503
title: "Collections drops a pane toggle that arrives before the shell is measured, and a hard-delete refused with revision_conflict never reloads the detail"
status: Done
assignee:
- '@claude'
updated_date: 2026-11-18 07:11
created_date: 2026-11-17 05:27
labels:
- follow-up
- library
- collections
priority: medium
references:
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_collections_controller.py
dependencies:
- TASK-41126
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41126 (libcoll, round 15; both also on base). (1) `LibraryScreen._toggle_library_media_reader_pane` (`library_screen.py:7919`) drops a Collections pane toggle (Escape or the grip) that arrives before the shell is measured, so the first key press does nothing. (2) A Collections hard-delete refused with `revision_conflict` only notifies; it never reloads the detail, so a retry is refused again until the user reselects the capture (code reading, not reproduced live). The saved-search `show_page` overlap with the pane-sync lock was fixed in 41126's fix round and is not part of this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A pane toggle sent before the first layout measure is applied once the shell is measured, or is refused visibly
- [x] #2 A hard-delete refused with `revision_conflict` reloads the capture so the next Delete press works without reselecting
- [x] #3 Each behaviour has a test shown red before the fix
<!-- AC:END -->

## Implementation Plan

1. Reproduce both at the base: a Collections pane toggle posted while the shell's allocation is not yet current, then the measured sync; and a hard-delete confirm refused with `revision_conflict` followed by a second confirm.
2. Write one real-screen test per behaviour in `Tests/UI/test_library_collection_reader_journeys.py` and show each red.
3. Toggle: when the Collections layout sync cannot measure, keep the toggle's priority on the stored layout so the measured sync applies it (the resolver already inherits `previous.priority_pane`).
4. Hard delete: on a `revision_conflict` refusal, reload the selected capture's detail (`refresh_selected_detail`) so the still-armed confirm sends the current revision.
5. Lane the new tests (census tail, `MINIMUM_FILES`), run covering tests, ratchets and preflight.

## Implementation Notes

- **Toggle: what was dropped.** At 80x24, Collections opens on its reader stage: Library is closed while its preference says open. A grip press, or Escape (`action_library_narrow_stage_return`), therefore reopens Library through `priority="library"` alone, because the preference it writes is already "open". `_sync_library_collections_reader_layout_from_shell` returned early while the shell's allocation was not current, and that discarded the priority. The measured sync then resolved the stage again, so the press did nothing. A probe at 170x48 showed the other case was already fine: there the toggle changes the preference, and the measured sync applies it.
- **Toggle fix.** When the sync cannot measure and was given a priority, it stores the priority as `priority_pane` on the Collections layout. The measured resolve already inherits `previous.priority_pane` while that pane's preference is open, so no new path was needed. This stores only what the sync would have applied had it measured.
- **Hard delete fix.** When a confirm is refused with `revision_conflict`, the controller warns as before, then reloads the selected capture (`refresh_selected_detail`, through `_run_library_collections_capture_transition` so the reader repaints). The confirm stays armed, so the next **Delete permanently** sends the current revision. A failed reload warns with its reason.
- **Tests** (`Tests/UI/test_library_collection_reader_journeys.py`, real Library screen):
  - `test_a_pane_toggle_sent_before_the_shell_is_measured_lands_once_it_is` gates `_library_adaptive_reader_allocation_is_current` to False at 80x24. It presses the Library grip and waits for the toggle's persistence claim, checks nothing applied yet, then lifts the gate and posts the shell's own `AdaptiveReaderShellResized`.
    - Red: `The early toggle was dropped: AdaptiveReaderEffectiveLayout(library_open=False, items_open=False, ..., reader_width=70, priority_pane=None, ...)`.
    - Green with the fix.
  - `test_a_hard_delete_refused_as_stale_reloads_so_the_next_confirm_deletes` moves the open capture on behind the reader (`update_capture`), presses confirm and waits for the `revision_conflict` warning. It then waits for the reload and presses confirm again.
    - Red: `The refused delete never reloaded the capture`.
    - Negative control, without the fix and without the reload wait: `The next confirm did not delete: ['revision_conflict', 'revision_conflict']`.
    - Green with the fix.
  - Both passed 8 of 8 parallel runs at load 18-28.
- **Lane.** Census tail, `MINIMUM_FILES` 2250 -> 2255, shared with TASK-41502. Serial lane-flag times at load ~15: toggle 1.6 s, delete 4.0 s.
- **Covering runs.** The collection reader, browse, compact and saved-search journeys, the capture reader, characterization, capture controller and reader geometry files, `test_library_crit8_collections_row.py` and `test_library_collections_wiring.py`, at `-n 4`, load 19-28: 73 passed, 53 failed.
  - All 53 are the local `RecoveryRequired: raw_source_selection_changed` trap on tests without a profile marker.
  - All 53 fail the same way with the base source restored.
- **Architecture, serial.** The size ratchets and the Collections and Prompts wiring tests: 93 passed.
- **Size.** `library_collections_controller.py` 1756 -> 1771 (measured). The pin was raised with a dated comment.
- **Docs.** The `Docs/User_Guide/library/collections.md` row for Delete Permanently… now says what a refused delete does.
- Files: `tldw_chatbook/UI/Library_Modules/library_collections_controller.py`, `Tests/UI/test_library_collection_reader_journeys.py`, `Tests/Architecture/test_library_modules_size_ratchet.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `Docs/User_Guide/library/collections.md`.
