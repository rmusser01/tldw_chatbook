---
id: TASK-42469
title: "Console agent-rail tests on no PR lane: the sub-agent badge's child case is red on dev, and the agent rail's full-log probe tests flake locally"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-24 00:57
labels:
- follow-up
- console
- agents
- tests
- dev-red
priority: medium
references:
- Tests/UI/test_subagent_badge_primary_token.py
- Tests/UI/test_console_agent_rail.py
- Tests/UI/test_console_agent_controller.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), in TASK-42426's covering runs. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths start at the repository root.

**Why.** Neither file is in `scripts/ui_pr_gate_census.txt` or a `derived-artifacts.yml` step, so their reds and flakes report nowhere.

**Sub-agent badge (red every time).** `Tests/UI/test_subagent_badge_primary_token.py::test_real_badge_count_work_tracks_subagents_not_primary_publication[child]` fails at base and on TASK-42426's branch, and again on the round-22 stack before its rebase (142e821116, `-n 4`): `TypeError: a coroutine was expected, got <function _controller.<locals>.repaint ...>`. It is raised by the test's `schedule` double (`asyncio.create_task(awaitable)`, line 184), reached from `_settle` (line 213). The `_controller` double (line 178) defines `async def repaint()`, but `schedule` is now handed the function itself rather than a call of it. Whether the product's scheduling call or the double changed is not known. The other three cases (`primary_only`, `ttl` and `database_owner`) pass. The file last changed in 3e618d7119 (2026-10-15).

**Agent rail full-log probes (flaky locally).** `Tests/UI/test_console_agent_rail.py` (46 tests) fails locally on the known `RecoveryRequired: raw_source_selection_changed` trap, which is out of scope here. Under a temporary `bootstrap_profile` mark, TASK-42426 ran it with `test_console_agent_controller.py` and `test_console_fleet_historical_detail.py`: 12 failures on its branch and 13 at base, 10 of them the same. The rest were full-log probe tests in `test_console_agent_rail.py` (the "View full log" availability tests from line 846, such as `test_full_log_probe_is_cached_per_run_id_while_open` and `test_full_log_slow_probe_is_off_thread_and_rejects_stale_target`) whose failures changed from run to run: two more branch runs failed 11 and 9. The node ids of the 10 consistent failures were not recorded.

**Earlier record.** TASK-32470 (open, medium, 2026-09-11, "Fix dev-tip test harness drift in console suites") already lists `test_console_agent_controller.py`'s `test_agent_bridge_is_*` tests as harness drift (its SimpleNamespace lacks `get_message_exchanges` and `capture_console_settings_origin`). Some of the 10 consistent failures may be those; AC#3 names them either way.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The badge `[child]` red has a recorded cause (test-double drift after a named commit, or a product defect) and is fixed: a defect in product code with a test shown red first, drift in the test with its assertions kept
- [x] #2 The full-log probe tests that flake are named by node id, each with its race named and fixed where it lives (product race in product code, test synchronisation in the test), and they pass in repeated runs of the file under parallel load (at least 20 runs, or a soak recorded in the notes)
- [x] #3 The three files' consistent failures under the bootstrap mark are named by node id in the notes, each with a cause or with the reason it is an artifact of the mark
- [x] #4 The badge file and the agent-rail file run in a PR lane (their green nodes at least), with their serial time under the lane's flags recorded in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Badge `[child]`: confirm the `schedule` double is handed the repaint function (not a coroutine) since d043e22438, and make the double accept what Textual's `run_worker` does (a coroutine or an async callable); assertions unchanged.
2. Run `test_console_agent_rail.py`, `test_console_agent_controller.py` and `test_console_fleet_historical_detail.py` under a temporary module `bootstrap_profile` mark, repeatedly at -n 4 under load; name each consistent failure by node id with its cause (or why it is an artifact of the mark).
3. Find the full-log probe tests that flake, name each race with a forced-delay reproduction, and fix it where it lives (product race in product code, test synchronisation in the test), keeping assertions; soak 20 runs under load.
4. Lane the badge file and the agent-rail file (green nodes at least) at the UI census tail, timed serially with the UI lane's flags.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Badge `[child]` (AC#1): test-double drift after d043e22438** (2026-11-05, "Fix native Console sync coroutine worker admission"). The count publication (`UI/Console_Modules/agent.py`, `_console_subagent_counts_for_rows`' read) now hands `run_worker` the repaint coroutine function from `_sync_native_console_chat_ui_accessor()`, which Textual's worker calls, instead of a coroutine. The test's `schedule` double passed whatever it got to `asyncio.create_task`, so it raised `TypeError: a coroutine was expected`. The double now does what `run_worker` does: it calls a callable and schedules a coroutine as is. Assertions unchanged. `[child]` was red at base and passes alone and with its file (4 of 4).

**Full-log tests that flaked (AC#2), and their races.** Evidence: three `-n 4` runs of the three files under a temporary `bootstrap_profile` collection hook (load 21-25), a bounded repro loop with a temporary hook that named the workers any `wait_for_complete()` saw cancelled (load about 50), and the soak's first attempt. Every race is test synchronisation: after mount the Console keeps ticking, and its control-bar sync and tick projection (`_sync_console_rail_and_controls`) read the Agent section's lines and fleet rows, the rail state and the section sync from whatever the test had attached. All fixes are in the test file; the product behaves as designed.
- `test_negative_availability_retries_after_real_first_record_append`: failed 2 of 3 runs, `[False, False, ..., True] == [False, True]` and `[False, False] == [False]`. The tick's control-bar sync calls the screen's `_sync_console_agent_section`, which the test had replaced with its publication recorder. It also failed once in the repro loop with `WorkerCancelled` (below).
- `test_full_log_probe_is_cached_per_run_id_while_open`: 1 of 3, `TypeError: <lambda>() got an unexpected keyword argument 'inspector_state'` from `_sync_console_control_bar_under_config` (`chat_screen.py:20973`), which asks the rail state with keywords; the test's stub took none.
- `test_initial_empty_log_semantics[absent]` and `test_initial_page_rejects_target_change_without_availability_scan[False]`: 1 of 3 each, `'Bridge' object has no attribute 'live_snapshot'` (`agent.py:1578`, `_console_agent_section_lines`), read by the tick from a partial bridge double.
- `test_view_full_log_no_ops_when_there_is_no_target_run` and `test_initial_empty_log_semantics[absent]` (first soak attempt, run 3 of each loop, load about 50): `WorkerCancelled` out of `host.workers.wait_for_complete()`. The repro hook named the cancelled worker: `_sync_console_legacy_workspace_context_aliases`, group `console-workspace-context-legacy-aliases`, which the Console restarts as an exclusive worker after each workspace-tray recompose; Textual's wait raises for the replaced one.
- `test_full_log_slow_probe_is_off_thread_and_rejects_stale_target` (named in the filing) did not fail in these runs; it has the recorder and partial-double exposures and is fixed with them.

Forced reproductions (temporary edits, reverted): one `console._sync_console_rail_and_controls()` right after the double was attached made `test_initial_empty_log_semantics` fail 3 of 3 with the `live_snapshot` error, the cached-probe test fail with the `inspector_state` TypeError, and the negative-availability test fail with `[False, False] == [False]`. Restarting the alias worker's group during the wait made the old `wait_for_complete()` fail with `WorkerCancelled` and the fixed wait pass.

Fixes (`Tests/UI/test_console_agent_rail.py`), assertions kept:
- `_RailBridge`: the quiet surface the tick reads (`subagent_counts`, `live_snapshot`, `subagent_run`, `subagent_runs`). Every bridge double in the file subclasses it (26). With the forced tick inserted after every attach (37 places), the file passed 51 of 51; before every double used the base, the same check failed 7 more tests on `subagent_counts`.
- `_rail_state_with_agent`: the real rail state with `agent_open` forced, taking the product's keywords.
- `_record_probe_publications`: records the section syncs made inside `_publish_console_agent_log_availability` only, as `test_old_log_probe_cannot_settle_newer_pending_generation` already wrapped the publisher. Used by the four tests that counted publications. Negative control: a temporary product edit that publishes twice still fails all four (`[True, True] == [True]`, `[False, False] == [False]`).
- `_settle_workers`: waits for each app worker and treats `WorkerCancelled` as settled; `WorkerFailed` still fails the test. It replaces the file's 13 `wait_for_complete()` calls.

**Consistent failures under the bootstrap mark (AC#3).** In all three runs:
- `test_console_agent_rail.py::test_modal_log_predicate_uses_turn_token_without_metadata_reads` (fixed here): it clicked Next as soon as the modal entered the screen stack. At that moment the modal was not mounted, Next's region was empty, and `pilot.click` returned False, so the second page never loaded and the "SECOND" wait timed out. It now waits until Next has a region and asserts the click landed. The modal itself appeared in 0.04 s.
- `test_console_agent_controller.py::test_persisted_run_state_reaches_the_mounted_agent_rail_statics` and `::test_agent_section_sync_skips_repainting_an_unchanged_payload`: `'Agent: idle' == 'Agent: done'`. Drift after 3e618d7119 (2026-10-15), which moved the historical snapshot read off the loop (`_presentation_historical_snapshot` returns an empty snapshot until the read lands); the tests read the lines synchronously.
- `test_console_agent_controller.py::test_drilldown_row_click_retargets_the_full_log_to_that_run`: `AttributeError: _console_agent_full_log_run_id`. Drift after 0436923835 (2026-09-12), which replaced it with the probe cache (`_console_agent_full_log_cache_run_id`).
- `test_console_agent_controller.py::test_subagent_badge_counts_batch_once_and_cache_until_the_row_set_changes`: `{} == {'conv-A': 2, 'conv-B': 1}`. Drift after 3e618d7119, which moved the badge count read to a worker (the first call returns `{}` and schedules it).
`test_console_fleet_historical_detail.py` had no failure under the mark. The four controller nodes are outside this task's lanes and are not repaired here; TASK-32470 (open) already holds that file's harness drift. Without the mark, the agent-rail file's mounted tests stop at `RecoveryRequired: raw_source_selection_changed`, the known trap.

**Lanes (AC#4).** The agent-rail file gains a module `pytestmark = pytest.mark.bootstrap_profile`, as the badge and composer files already have, because its mounted tests read config and a sandboxed lane stops them at the trap (a plain gated UI test that reads config fails in CI too, per TASK-34982's lesson). Coordinator ruling (2026-11-23): keep the module-level marker. It follows TASK-41313's precedent, and it is an engineering call, not an owner decision. Both files join the UI census tail (the agent-rail file was later split into node ids; see "Census split" below). Serially with the UI lane's flags: badge 9 passed in 19.6 s (call+setup+teardown 11.8 s) at load 41; agent rail 51 passed in 330 s at load 49-53 (286 s of setup+call+teardown, about 7 s per mounted test; 326 s at load 43-57 on the previous revision). The census floor is the measured entry count (shared with TASK-42467's 13 entries).

**Soak (AC#2).** The final file, 20 whole-file runs at `-n 4` as two concurrent loops of 10, beside other agents' load (1-minute load average 56-98): 20 of 20 passed, 51 of 51 each, 98-150 s per run. Before the `_settle_workers` fix, the first soak attempt failed 2 of 6 runs (the `WorkerCancelled` cases above) and was stopped. Together with the badge file at `-n 4`: 60 passed (load 37).

**Files.** `Tests/UI/test_subagent_badge_primary_token.py`, `Tests/UI/test_console_agent_rail.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, and a `backlog/docs/lessons-testing-evidence.md` entry (doubles attached to a mounted Console are read by its own tick; force one tick to reproduce).

**Census split (review, 2026-11-23).** The agent-rail file is no longer one whole-file entry (330 s), which `--rebalance` could not split across shards. It is now one node id per test function, 46 functions as 48 entries: `test_initial_empty_log_semantics` (36.6 s in all) is listed by case, and every other function is at most 27.1 s. The badge file stays one whole-file entry (9 tests, 19.6 s; 11.8 s setup+call+teardown). Serial seconds per entry (setup+call+teardown) with the UI lane's flags (`-p no:xdist --timeout=180 --durations=0`), from one run of the file at load 70-106 (51 passed in 433 s): `test_conversation_row_carries_badge_count_for_render` <0.1; `test_conversation_row_badge_label_is_escaped_and_present` <0.1; `test_no_badge_when_subagent_count_is_zero` <0.1; `test_badge_renders_on_its_own_line_regardless_of_secondary_length` <0.1; `test_wrapped_title_still_pairs_with_full_badge` <0.1; `test_short_title_without_badge_is_unchanged` <0.1; `test_conversation_row_height_tracks_name_lines_and_badge` <0.1; `test_agent_rail_section_mounts` 11.5; `test_resume_rederives_subagent_data_from_durable_run_store` <0.1; `test_bridge_subagent_counts_batches_in_one_db_call` <0.1; `test_subagent_counts_are_batched_and_gated_not_refreshed_every_tick` 12.1; `test_an_active_run_requeries_subagent_counts_at_most_once_a_second` 9.9; `test_summarize_step_does_not_escape_markup_brackets` <0.1; `test_spawn_subagent_summary_does_not_escape_markup_brackets` 2.0; `test_agent_section_lines_render_brackets_literally_not_escaped` 6.8; `test_agent_section_falls_back_to_historical_snapshot_when_live_is_idle` 9.2; `test_agent_section_prefers_live_snapshot_over_historical_when_present` 8.8; `test_resume_rederives_top_level_agent_summary_from_durable_run_store` <0.1; `test_drilldown_falls_back_to_overview_after_conversation_switch` 7.3; `test_drilldown_render_path_rejects_record_from_other_conversation` 9.0; `test_drilldown_step_text_grows_with_a_configured_cap_above_eighty` 9.5; `test_full_log_probe_never_touches_the_bridge_while_collapsed` 7.9; `test_full_log_probe_is_cached_per_run_id_while_open` 9.9; `test_view_full_log_loads_off_thread_then_opens_the_modal` 10.4; `test_view_full_log_no_ops_when_there_is_no_target_run` 9.9; `test_activate_native_console_session_clears_stale_drilldown` 8.1; `test_clicking_a_specific_subagent_row_drills_into_that_run_directly` 9.2; `test_console_agent_fleet_token_total_sums_live_handles` 7.7; `test_console_agent_fleet_token_total_is_zero_with_no_bridge` 8.1; `test_cancel_console_agent_fleet_row_delegates_to_the_bridge` 8.3; `test_cancel_console_agent_fleet_row_returns_false_with_no_bridge` 7.8; `test_cancel_console_agent_fleet_row_returns_false_with_no_row_id` 8.9; `test_drilldown_shows_a_live_childs_steps_before_they_reach_the_db` 8.6; `test_drilldown_still_prefers_the_persisted_steps_once_they_exist` 8.3; `test_full_log_slow_probe_is_off_thread_and_rejects_stale_target` 14.3; `test_negative_availability_retries_after_real_first_record_append` 12.7; `test_pending_negative_probe_never_restarts_and_updates_real_affordance` 10.9; `test_initial_page_rejects_target_change_without_availability_scan` 26.4; `test_initial_empty_log_semantics[absent]` 11.9; `test_initial_empty_log_semantics[empty-end]` 10.6; `test_initial_empty_log_semantics[empty-continuation]` 14.2; `test_log_probe_settles_stale_or_cancelled_generation` 25.7; `test_log_target_metadata_resolution_is_off_thread_and_stale_safe` 27.1; `test_log_target_metadata_and_binding_token` <0.1; `test_log_path_does_not_initialize_an_absent_runtime` 10.8; `test_old_log_probe_cannot_settle_newer_pending_generation` 12.7; `test_modal_log_predicate_uses_turn_token_without_metadata_reads` 13.6; `test_log_probe_cancelled_before_entry_retries_on_existing_tick` 11.5. The census floor is the measured entry count, 2597 (67 tail entries from this branch, with TASK-42467's composer entries).
<!-- SECTION:NOTES:END -->
