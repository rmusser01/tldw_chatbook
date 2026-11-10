---
id: TASK-39170
title: "test_destination_visual_parity_correction.py fails under its plain command and runs on no PR lane, and under a profile marker 25 of its nodes are red on their own assertions on dev"
status: Done
assignee:
- '@claude'
created_date: 2026-11-08 06:54
updated_date: '2026-11-10 20:35'
labels:
- follow-up
- tests
- test-infra
- ui
priority: low
references:
- Tests/UI/test_destination_visual_parity_correction.py
- Tests/UI/app_factory.py
- Tests/conftest.py
dependencies:
- TASK-2560
- TASK-2960
- TASK-38111
- TASK-39134
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (settings stream, TASK-39134's notes, round-12 branch fu/r12-settings). Checked on dev b590a15afd, 2026-11-08 (run.sh, `-n 4`, load ~21-25).

**The file.** It has 70 test functions. Three are `@private_profile_test`, none carries `bootstrap_profile`, and no PR lane or census entry runs it.

**Plain.**
- 113 nodes fail with `RecoveryRequired: raw_source_selection_changed`, through `_build_test_app`.
- 2 fail on their own: `test_library_source_snapshot_times_out_to_stable_error` and `test_library_source_snapshot_timeout_handles_blocking_async_services` (`'types.SimpleNamespace' object has no attribute 'app_config'`).

**Under a temporary `bootstrap_profile` marker** (scratch plugin), 25 nodes are red on their own assertions:
- 6: a mode or category strip 18 rows tall (`#artifacts-mode-strip`, `#acp-mode-strip`, `#settings-category-strip`: "is too tall: Region(x=0, y=5, width=140, height=18)").
- 4: `KeyError: 'mcp'` and `KeyError: 'skills'` in the compact-size and tab-order matrices.
- 7: Watchlists. `NoMatches '#watchlists-detail-pane'`; `NoMatches '#sources-type-select'` (2); the selected row not painted in reverse video (2); "Timed out waiting for ChatScreen" (2).
- 4: copy pins. `'Last pull: —'` (Schedules) and `'Run unavailable'` (Workflows), plus the operational loading-state matrix for each.
- 1: `test_core_default_empty_or_blocked_states_keep_workbench_geometry[chat-…]` (`assert 30 < 0`).
- 1: `test_settings_boundary_note_is_fully_visible_at_rest[size0]` (140x42), "boundary note Region(x=106, y=37, width=31, height=3) outside the inspector window Region(x=106, y=19, width=31, height=18)". It is red on the round-12 stack too.
- 2: the two snapshot-timeout nodes above.

**Round 12 and open tasks.**
- TASK-39134 (round 12) rewrote one pin in this file on purpose: `test_settings_inspector_floor_yields_to_the_detail_pane_when_narrow`, 140 → 150 columns.
- TASK-39134 also saw five Settings nodes red on both trees: the two strips, the 140x42 boundary note, and `KeyError: 'mcp'` / `'skills'`.
- Open tasks already cover some of the file's nodes: TASK-2560 (Schedules nodes and a missing media DB) and TASK-2960 (MCP forced loading).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The file's nodes run under its plain command without `RecoveryRequired`, through the suite's existing profile seams; no test binds config to the real profile, and RecoveryRequired admission is unchanged
- [x] #2 Each own-assertion red is classified as test drift or a product defect, and fixed or retired with the commit that moved the behaviour named; a product defect (such as the boundary note outside the inspector at 140x42) gets a test that is red before its fix
- [x] #3 Overlap with TASK-2560 and TASK-2960 is reconciled in the notes
- [x] #4 The file runs on a PR lane (never the fast PR contract), split as its timing needs, with three timings recorded
<!-- AC:END -->

## Implementation Plan

1. Add the module `bootstrap_profile` marker (TASK-39141's seam) so the file's nodes run plain without `RecoveryRequired`; keep the three `@private_profile_test` nodes as they are.
2. Run the file at base under the marker; for each own-assertion red, find the commit that moved the behaviour (git log -S / blame) and classify it as drift or a product defect.
3. Drift: re-pin to the shipped behaviour, or retire naming the commit that removed the subject. Product defect: fix the product with this file's test red before the fix.
4. Reconcile overlap with TASK-2560 and TASK-2960 in the notes.
5. Time the whole file serially three times (the UI lane's command) and hand the per-test durations to the lane split.

## Implementation Notes

**Profile seam (AC#1).** Module `pytestmark = pytest.mark.bootstrap_profile` (TASK-39141's seam); the three `@private_profile_test` nodes keep their children. No conftest, `private_profile.py`, `app_factory` or admission change. Base ae9f9e8d21, plain, run.sh, `-n 4`: 116 failed / 4 passed / 1 error, 113 of them `RecoveryRequired: raw_source_selection_changed` (load 39-55). Head, same command: 115 passed / 1 failed, 0 `RecoveryRequired` (load 37-50); the one red is `test_library_source_browser_collections_action_switches_to_collections_mode`, below.

**Own reds (AC#2).** Base under the marker: 27 failed / 93 passed / 1 teardown error (load 43-53). Each, with the commit that moved it:
- Drift, harness CSS (6: the three `source_prep[artifacts]` nodes, `runtime_and_settings[acp|settings]`, `acp_runtime_blocked`): ae9093a714 replaced DestinationModeStrip's inline one-row height with the bundle's `.h-1`, and these mounted without the bundle (strip 18 rows). They now mount under the production stylesheet set (artifacts joins `_visual_destination_harness`'s production map; the two others use `_ProductionDestinationHarness`); the 2-row strip pin is unchanged.
- Retired (4: `compact[mcp|skills]`, `tab_order[mcp|skills]`): 42a5106c41 (task-35150) retired the MCP screen into Settings > Tools & MCP, d72984cec6 (task-34893) the Skills screen; both left SCREEN_BY_ROUTE (KeyError). The three MCP tests here measure the Tools & MCP pane.
- Drift, stand-ins (2 snapshot-timeout nodes): 11b2fa7005 made `LibraryScreen.__init__` read `app_instance.app_config`; the SimpleNamespace now carries `app_config={}`. The blocking-services node then measured 0.26 s: 469935eabc stopped isolating the conversations call in a worker because the real scope service offloads its own SQLite read (task-283), so its stand-in now awaits instead of blocking the loop; notes and media still block and stay isolated.
- Drift, copy (2): Schedules local-only collapse (683a73c029, task-23105) hides the owner buttons and pull/push stamps behind "Local schedules — no scheduling server connected; sync is off."; Workflows execution (12a15a36ea) reads "Run a saved revision · …" and "No existing Console handoff". Re-pinned to the new copy plus the absence of the old.
- Drift, Watchlists (5): `compact[watchlists_collections]` — 04222fe8eb widened Feed Items' floor 32 -> 42, so Read parks it at 100 columns: detail is now the permanent Reader and the `#wl-grip-items` grip is asserted visible. Sources toolbar (2) — 3c1b55f3ca moved Type/Status/Active into the Filters editor: the toolbar row and, after Filters, the editor row are each checked for containment and painted labels. Tree selection — Read is the landing tab (66682cf786) and since 0bde972ca8 a Read tree click commits only with a Feed Items snapshot, which this static harness never serves: the test runs on Overview, where it was written. Sources reverse video — `_row_reverse_video` matched the inspector's "Selected: AI News RSS" (5c5d3a1c56) on the table's header row; rows are now cropped to the table's columns (the selected row does paint reverse).
- Test races (6): Console rail (`core_default[chat]`, `assert 30 < 0`) waited for `display`, which the open sets before layout: waits for its region. Chevrons (2): boot to ChatScreen took ~7.4 s at load ~60, past the 6 s default: 30 s, as the MCP forced-loading test. `settings_inspector_rows[*]` (load-only): two pauses vs the async pane swap (4fe45538f5): waits for the swap like the shared press helper. Pane titles `[workflows]` and the two MCP Tools-pane nodes (seen in 1 of 3 serial runs, load ~45-80): wait for the editor heading's first load and the canvas's layout. The `service_call_awaits…` teardown error: its global `asyncio.to_thread` patch outlived the call into 5f00aa5929's runtime drain; now scoped with `monkeypatch.context()`.
- Product defect, Settings: `test_settings_boundary_note_is_fully_visible_at_rest[140x42]` red at base (note y=38..40, window 19..36, the 34-column inspector of TASK-39134's ruling). The Boundary row now leads the inspector's rows (`settings_screen.py`); green after, the 211x44/235x52 cases and the floor/reachability/action-row nodes still green. User Guide (settings.md, Scope Inspector row) updated.
- Product defect, Schedules: `test_operational_loading_states_preserve_workbench_geometry[schedules]` red at base: before the first `set_task` TaskDetail showed its blank form, so during the queue load Follow in Console sat 46 rows into a 20-row pane at 140x42. TaskDetail now composes in the empty state `set_task(None)` paints (`task_detail.py`); green after.
- Not fixed: `test_library_source_browser_collections_action_switches_to_collections_mode` failed 3 of 5 whole-file runs here (passed in 2 subset runs) with "Library source services unavailable; … an unexpected error"; not in the filed 25, cause not read.

Covering runs for the two product changes (temporary marker plugin, deleted): 11 Schedules/Settings files, 26 failed / 302 passed at head; those 26 re-run with base product sources: 21 failed; the 5 head-only reds passed 3 of 5 on a head re-run, the other 2 failing with worker `NoMatches` races (DefinitionDetail, task table) of the base class. `Tests/Architecture/test_module_size_ratchet.py` 201 passed (settings_screen.py at its base line count).

**Overlap (AC#3).** TASK-2560's four Schedules media-DB cases pass under the marker here; the "Local media DB is required" ValueError is still logged from `_refresh_console_context`, but no node fails on it, and the loading-state red here was the TaskDetail form, not the media DB. TASK-2960's `test_mcp_forced_loading_state_stays_inside_workbench` passed in every run here (MCP now lives in Settings > Tools & MCP, 42a5106c41). Both look closable; the coordinator decides.

**Timing (AC#4 input; lane not added here).** Serial, the UI lane's command (`--timeout=180 --durations=0 --tb=short`): 589 s (load 29-54), 741 s (load 29-46), 850 s (load 43-46). The file is too heavy for one whole-file census entry; the two Settings action-row and inspector-reachability functions alone are 142 s and 108 s at their slowest.

Files: `Tests/UI/test_destination_visual_parity_correction.py`, `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/UI/Screens/scheduling/task_detail.py`, `Docs/User_Guide/settings.md`.

**Lane (AC#4).** The file is laned by function: 69 per-function selectors at the UI census tail, charged at each function's slowest serial seconds (about 893 s in all). `--rebalance 13.3` then swapped 6 pairs, and the max shard is 13.1 pytest min over the 22 cached PR runs. The file's whole-file serial runs were 589 s, 741 s and 850 s.

The Collections action node was left out at first as a flake. It is now laned too; its cause and its 10-of-10 runs are under "Review r13c-rev1" below.

The covering run for the Settings change (137 census entries that touch the Scope Inspector, `-n 4`, load 40-50): 229 passed, 2 failed. Both failures (`test_add_to_model_list_names_what_it_added_or_how_to_choose_a_default[added-then-already-listed]`, `test_settings_keyless_provider_save_keeps_template_env_name_out[legacy-env-unset]`) then passed 2 of 2 alone at head and 2 of 2 with base `settings_screen.py`, so they are load flakes, not this change.

**Review r13c-rev1.**
- **I2, the Collections action node (harness timing).** Its cause is now read, and it is laned.
  - The factory app opens no ChaChaNotes DB (`_open_profile_chachanotes` is patched to None), so `chat_conversation_scope_service` has no local service. The Library source snapshot therefore fails with "Chat conversation local service is unavailable." in passing and failing runs alike (4 of 4 with `-rA`), so the snapshot failure is not the flake.
  - What fails is `_wait_for_selector`'s 2 s default. Timed with a 30 s wait (temporary), `#library-collections-reader-shell` mounted after 2.67, 2.77, 2.52, 2.67, 2.79 and 2.51 s at load ~46, every run.
  - Red with the default: 2 of 4 alone. The wait is now 15 s, and the shell must still mount. Green: 10 of 10 alone at load 42-53.
  - Not a product race: the shell always mounts, only later than 2 s under load.
  - The node joins the UI census tail; the floor is now 1985, and the replay's max shard is still 13.1 min.
- **(b) TaskInspector, the sibling of the TaskDetail defect.** Its metadata rows and its "No conflict" card now compose hidden, the empty state `set_task(None)` paints.
  - New `Tests/UI/test_schedules_ux_fixes.py::test_detail_and_inspector_open_in_their_empty_state_before_any_task`, a whole-file census entry, is red before the change (`assert not True` on `#scheduling-inspector-metadata`) and green after (the file gives 10 passed).
- **Covering run** (13 Schedules files, temporary bootstrap marker plugin, deleted): 19 failed / 327 passed. With base `task_detail.py` (ae9f9e8d21), the same 19 gave 17 failed, 2 passed. The three nodes red at head and green at base:
  - `test_returning_to_the_screen_while_stale_refetches_definitions` is red on both trees on rerun.
  - `test_runs_on_dropdown_confirm_dialog_lists_warnings` is intermittent on both.
  - `test_at_the_floor_the_queue_is_full_width_and_the_rows_paint` paints only chrome when it fails. In interleaved runs: head 2/8 vs TaskDetail-only 3/8, then head 0/12 vs base 2/12 (load 39-53), so it is a pre-existing load flake.
- **M5.** The User Guide sentence now says the Boundary note (where this page's responsibility ends) comes "ahead of the ownership rows". It no longer promises a view without scrolling.
