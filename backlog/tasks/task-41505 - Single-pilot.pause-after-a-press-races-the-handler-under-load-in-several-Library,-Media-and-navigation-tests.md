---
id: TASK-41505
title: "Single pilot.pause after a press races the handler under load in several Library, Media and navigation tests"
status: Done
assignee:
- '@claude'
updated_date: 2026-11-18 20:54
created_date: 2026-11-17 05:27
labels:
- follow-up
- tests
- flaky
priority: medium
references:
- Tests/UI/test_library_collections_actions.py
- Tests/UI/test_screen_navigation.py
- Tests/Library/test_library_ingest_runner.py
dependencies:
- TASK-41127
- TASK-41126
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Each of these waits one `pilot.pause()` after `Button.press` (or after a flag) and fails at load about 65-110 while passing at low load. Found in round 15:
- libcoll: `test_quick_capture_refresh_hard_delete_cancel_and_legacy_recovery_close` (characterization test), and the open-original check (the handler runs on a background thread);
- libtests review: `test_analyze_bulk_action_follows_the_selection_in_place` and `test_pressing_analyze_leaves_select_mode_and_paints_its_receipt` (render_fixes, off lane), `test_info_describes_the_same_stored_text_representation_sent_to_console` and other press-after-flag media nodes;
- libtests: existing census nodes flaky at load about 100: `test_navigation_keypress_during_splash_is_safely_ignored`, `test_boot_with_mcp_default_tab_lands_on_settings_tools_hub`;
- endpoint fix r1: `test_actual_handler_review_wake_waits_for_pinned_capacity_and_finishes` failed 2 of 2 at load 65-110 but 4 of 4 passed at load 8-40 (timing, unproven);
- `Tests/Workspaces/test_native_confirmed_undo.py` fails at base and head on different nodes each run ("timed out waiting for ..." and `'after\n' == 'before\n'`); TASK-40716 only notes it.
- closeout final checks (2026-11-17, load about 40): `Tests/Library/test_library_ingest_runner.py::test_explicit_parakeet_directory_is_snapshotted_before_dispatch` failed once in the serial recovered-suites step; it waits one `pilot.pause()` after `submit_library_ingest_job` and then asserts the executor was called. It passed alone and in two whole-file runs at load 32-46, and no round-15 change touches the ingest runner or its tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each listed test waits on its observable condition (a state predicate with a bounded deadline), not on a fixed pause count
- [x] #2 Each passes 10 of 10 at `-n 4` with the load recorded in the Implementation Notes
- [x] #3 `test_actual_handler_review_wake_waits_for_pinned_capacity_and_finishes` has its cause identified, as timing or as a defect
- [x] #4 `Tests/Workspaces/test_native_confirmed_undo.py` passes 10 of 10 at `-n 4`, with the cause of each of its two failure shapes named
<!-- AC:END -->

## Implementation Plan

1. Scope: every listed node except the two `Tests/UI/test_screen_navigation.py` nodes (another stream owns that file).
2. For each node, find the fixed pause (or count-bounded wait) whose read races a handler, worker or recompose; replace it with a bounded wait on the condition the next assertion reads, with a message that names the state.
3. Where the press itself can race a recompose, wait for the live widget the press needs (TASK-41532/TASK-41703: a pending press can be late or lost; no blind re-press).
4. Grep each touched file for the same shape (press or submit, one pause, then a read) and fix or list each site.
5. Red before: a temporary delay in the raced work (reverted) turns the old test red on the line the fix changes; the new test is green under the same delay.
6. AC#3: force the suspected race in the review-wake node and name its cause (timing or defect). AC#4: name the cause of both `test_native_confirmed_undo.py` shapes and fix them.
7. 10 of 10 at `-n 4` per node with `uptime` load recorded; preflight and the size ratchets.

## Implementation Notes

- `Tests/UI/test_library_shell.py::test_library_media_edit_save_stays_gated_through_detail_refresh` had this cause (it waited for the interlock flag, then asserted the page and facet refresh calls its release starts on workers) and was fixed under TASK-41532.

**Round 16 (fu-r16-waits).** AC#1 and AC#2 are met for every node except the two `test_screen_navigation.py` nodes, which another stream owns; they stay unticked, and the task In Progress, until that stream lands. AC#3 and AC#4 are met. Test-only. Every listed node except the two `Tests/UI/test_screen_navigation.py` nodes (another stream owns that file) now waits for the state its next step reads. Each wait has a bounded deadline and a message that names that state. No cause was a product defect; the product limit behind TASK-41517 is separate. Each red below comes from a temporary edit, since reverted; old copies of the test files were run from scratch copies, since deleted. The plain Collections characterization file ran under a collection-time `bootstrap_profile` stamp from a throwaway plugin (the TASK-40328 recipe, for the local RecoveryRequired trap), with no edit to the file.

**Collections characterization** (`test_library_collections_characterization.py`).
- **Open original.** The handler opens the URL on a thread worker, so the test now waits for `opened_urls`. Red: a 1 s sleep in that worker; the old test failed `[] == [url]` (line 507), the new one passed.
- **Filters toggle** (found by the grep, in `test_capture_filters_...`). It read `filters_open` one pause after the press; it now waits for it. Red: a 0.5 s delay in the screen's toggle handler; old `False is True` (line 187), new green.
- **Quick Capture Cancel.** It now waits for the form to close (the flag, and the URL field gone).
- **Settles.** Seven settle loops before the next press (`range(5)` and `range(3)` pauses) became `_wait_for_capture_panes`: every mounted pane's `presentation` equals the current one and none is half-built, the same test `recompose_capture_panes` uses.
- **No red for Cancel or the settles.** A 0.5 s delayed Cancel (6 of 6) and a 0.4 s delayed pane sync (8 of 8) left the old test green. They were changed only so that no fixed pause count remains (AC#1). This test's two earlier failing checks were fixed by TASK-41126.

**Media render fixes** (`test_library_media_render_fixes.py`).
- **`test_pressing_analyze_...`.** The test pressed Analyze one pause after two row presses. Analyze stays disabled until the toggles land, and a press on a disabled Button is a silent no-op. The test then read the toolbar after one more pause.
  - Now: it waits for 2 checked rows and Analyze enabled. It holds the run's partition pass (`_library_media_unanalyzed_ids`), waits for the toolbar to drop while that is held, then releases it.
  - Red: a 0.5 s async delay in the media row handler. Old: `assert not <DOMQuery '#library-media-select-actions'>` at line 759. New: green.
  - Negative control: the handler's own `_sync_library_canvas(self, "media")` removed. New: red, "Analyze never dropped the select-mode toolbar." Old: green, because its one pause also passed on the worker's post-partition sync. The old test never pinned "at once".
- **`test_analyze_bulk_action_follows_...`.** The test now waits for `row_selection.count`. The toggle and the in-place patch run in one handler call, so the flip is checked at once, with no wait that a later recompose could satisfy.
  - Red: the same row delay; old failed "checked rows must arm Analyze" (line 911), new green.
  - Negative control: the in-place `analyze_button.disabled` patch removed; new red at line 951.

**Media reader flow, `test_info_describes_...`** (press after a flag).
- It pressed Info right after `loaded_backing_id` was published. The loaded detail re-mounts the Reader chrome after that flag, so it now presses the live tab from `_wait_for_selector`, as TASK-41127 did for its sibling.
- Red: the post-load `_recompose_library_media_detail_if_unrendered` deferred 0.5 s. Old: `NoMatches '#library-media-reader-select-info'`, 3 of 3. New: 3 of 3 green.
- A grep of `test_library_media_reader_flow.py` and `test_library_media_render_fixes.py` for a press straight after a flag wait found no other site. Keyboard `pilot.press` -> one pause -> assert sites remain in both files (for example reader_flow 134, 267, 307, 319; render_fixes 225, 1048, 1064, 1156, 1237). None was reported flaky; they are listed, not changed.

**Ingest runner** (`Tests/Library/test_library_ingest_runner.py`).
- A submit builds and sends its local STT dispatch on a `library-local-stt-dispatch-*` thread. That thread then blocks in `call_from_thread` until the submitted callback has bound the job on the loop.
- `_wait_for_dispatches` waits for the executor's calls (and parse submits, where the test asserts them) and for those threads to end. It is bounded at 10 s and names the counts and the live threads.
- It is used at the named node and at the 15 other submit -> one pause -> read sites in the file. Six of those were found only by the red run.
- Red: `time.sleep(0.5)` before `_build_local_stt_dispatch`. The old file failed 18 nodes, the named one included; the new file passed 156.

**AC#3, review wake** (`test_actual_handler_review_wake_waits_for_pinned_capacity_and_finishes`): timing, not a defect.
- **First cause: the waiter's deadline.** The transport's capacity-probe waiter (`observe_wait`) had a 15 s deadline that started before the first primary turn.
  - Red: a forced 1.0 s per POST made the first turn 18.7 s, and the waiter timed out. The probe hold was then never released.
  - Console then correctly refused the wake as `review_context_capacity_unknown`, and the test failed `[2] == [2, 1]` (3 of 3, load 13-19).
- **Second cause: the schedule driver's 200 x 0.03 s loop**, which gives up silently. With the waiter fixed, a 2.0 s per-POST delay still left the verifier unlaunched when the loop gave up.
- **Fix.** The waiter has no deadline of its own: the schedule wait bounds the run, and `close()` cancels the waiter. The driver loop and the journey's `_settled`, which had the same silent give-up loop, now wait on their conditions with 60 s wall-clock deadlines. They fail naming the counts, the delivering sessions and the terminal/idle state.
- **Results.** Green under both delays. 10 of 10 at `-n 4` (load 14-27). The three covering files passed 39 of 39.
- **Files.** `Tests/Chat/test_console_review_wake_transport.py`, `test_console_code_review_continuation.py`, `test_console_multi_model_review_journey.py`.

**AC#4, `Tests/Workspaces/test_native_confirmed_undo.py`**: two test-timing causes, no defect. At base, seven whole-file runs at `-n 4` (load 17-50) each failed 1 to 4 nodes.
- **"timed out waiting for ..."** (and its `pilot.click -> False` and empty-subprocess variants).
  - `pilot.click` aims at the region the button has when it is called. A just-mounted button has an empty region until the modal's next layout, so the click lands at (0, 0) and misses. A click probe over 3 base runs found 9 of 97 clicks missed, each with an empty region at call time.
  - Red: holding the modal's first layout for 1 s turned 30 of 33 old nodes red, and none of the new.
- **`'after\n' == 'before\n'`.** The revert worker starts in the modal's dismiss callback, which can run after the click returns. `workers.wait_for_complete()` then finds no worker and returns before the revert: the TASK-41513 trap.
  - Red: deferring `dismiss(True)` by 0.5 s turned 5 old nodes red, and none of the new.
- **Fix.**
  - `click_modal_button` clicks only a laid-out button and asserts the click landed.
  - `confirm_and_settle` waits for the callback's `confirm()` before it waits (30 s, named) on the workers. It waits on `confirm()` because `never_reselects[provider]` and `[capacity]` refuse at `confirm()` before any worker starts.
  - The subprocess cases print the child's receipt on failure.
- **Results.** 10 whole-file runs at `-n 4`: 330 of 330 (load 13-23).

**AC#2, 10 of 10 at `-n 4`.**
- The eight nodes changed here were run 10x each in one command: the three Collections nodes, the two Analyze nodes, Info, the parakeet ingest node, and TASK-41530's finishing card. The first batch passed 78 of 80; both failures were TASK-41530's node, whose notes cover them. The three identical batches after it passed 80 of 80 each (load 10.7-15.0).
- Review wake: 10 of 10 (load 14-27). Native undo: 10 whole-file runs, 330 of 330 (load 13-23).


**Fix round 1 (review r16wt-rev1).** `_wait_for_capture_panes`'s `stale()` used `query_one` for the Items and Work panes, so a pane caught mid-recompose raised `NoMatches` while the failure message was being built. It now uses `screen.query(...)`, as `recompose_capture_panes` does, and lists a missing pane as "<selector> missing", so a missing pane still counts as not painted.

**Covering runs** (`-n 4`). The render-fixes, reader-flow, ingest-runner, approval (three files), native-undo and review (three files) test files: 569 passed, 1 skipped (load 18-25). After the last edits, the ingest-runner, approval and Personal Context files: 417 passed, 1 skipped (load 28-30). The Collections characterization file: 5 passed. `Tests/Architecture/test_screen_size_ratchet.py`: 25 passed (serial). Preflight passed. Ruff reports nothing new against base in any changed file.

**Files.** `Tests/UI/test_library_collections_characterization.py`, `Tests/UI/test_library_media_render_fixes.py`, `Tests/UI/test_library_media_reader_flow.py`, `Tests/Library/test_library_ingest_runner.py`, `Tests/Chat/test_console_review_wake_transport.py`, `Tests/Chat/test_console_code_review_continuation.py`, `Tests/Chat/test_console_multi_model_review_journey.py`, `Tests/Workspaces/test_native_confirmed_undo.py`.

- The two `test_screen_navigation.py` nodes were fixed under TASK-41506 (fu/r16-nav). `test_navigation_keypress_during_splash_is_safely_ignored` waits for the splash to take focus, under a 120 s splash that only the key closes, and then for startup with `_settle`: red at base with a 1.6 s delay injected before the press (`splash must still be active at press time`), green after. `test_boot_with_mcp_default_tab_lands_on_settings_tools_hub` boots without the splash and waits for the initial screen, then up to 30 s for the hub: red at base with a 14 s delay injected into the hub load (`assert not ...is_loading`), green after. Each passed 10 of 10 at `-n 4`, load 11-15.

**Closed (round-16 coordinator, 2026-11-18).** The two `test_screen_navigation.py` nodes landed with TASK-41506 (fu/r16-nav, stacked below this commit), each 10 of 10 at `-n 4` at load 11-15 as recorded above, so AC#1 and AC#2 now hold for every listed node.
