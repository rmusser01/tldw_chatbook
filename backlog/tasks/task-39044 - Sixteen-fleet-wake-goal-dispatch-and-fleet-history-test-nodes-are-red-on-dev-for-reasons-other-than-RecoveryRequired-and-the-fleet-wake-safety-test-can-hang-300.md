---
id: TASK-39044
title: "Sixteen fleet-wake, goal-dispatch and fleet-history test nodes are red on dev for reasons other than RecoveryRequired, and the fleet-wake safety test can hang 300 s in its cleanup"
status: To Do
created_date: 2026-11-06 10:09
labels:
- follow-up
- tests
- console
- fleet
priority: medium
references:
- tldw_chatbook/Chat/console_fleet_wake.py
- Tests/Chat/test_console_fleet_wake_staleness.py
- Tests/Chat/test_console_goal_dispatch.py
- Tests/Chat/test_console_headless_wake_invariants.py
- Tests/Chat/test_fleet_wake_retry.py
- Tests/Chat/test_console_fleet_wake_safety.py
- Tests/UI/test_console_fleet_history_navigation.py
- Tests/UI/test_console_fleet_wake_ui_freshness.py
- Tests/UI/test_console_fleet_wake_wiring.py
- Tests/UI/test_console_headless_wake_fires.py
- Tests/UI/test_console_launch_wake.py
- Tests/UI/test_console_automatic_work_pause.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on the round-10 Console PRs (send #3433, head e3106a313c, which lands TASK-38971; Console tests #3434, head 16c99d2d9f), run stacked with storage #3435. Results are the same on both. Line numbers are dev's; every cited line is the same on those heads.

**How the reds split.** TASK-38971's 28-file fleet-wake set (`-n 4`, load 11-24) gave 62 failed / 177 passed on dev and 62 failed / 178 passed on the PRs.
- Rerun under a temporary `bootstrap_profile` marker (applied as a `-p` plugin, no tree edit), the 15 files with reds give 16 failed / 99 passed in both trees.
- So 46 of the 62 are TASK-38111's `RecoveryRequired: raw_source_selection_changed`. That is 30 directly, and 16 that read as "wake never delivered" when run plain but pass under the marker. TASK-38111 owns those 46.
- The other 16 fail with the marker too. No PR lane runs any of the 15 files; the one laned node, `test_automatic_launch_recovery.py::test_chat_opened_while_the_wake_loads_its_tree_is_not_opened_twice` (`derived-artifacts.yml:688`), passes.

**11 nodes red every run, in both trees:**
- `test_console_fleet_wake_staleness.py:119` `test_deferred_wake_notice_reports_done_despite_a_pinned_snapshot` ("clearing the draft never delivered the deferred wake").
- `test_console_goal_dispatch.py:102` `test_goal_native_dispatch_is_automatic_fresh_and_returns_exact_native_outcome`.
- `test_console_headless_wake_invariants.py:843` `test_a_headless_wake_takes_the_same_agent_dispatch_and_budget`. This is drift: the dispatch kwargs gained `on_primary_budget_settled` and `parent_budget_deadline`.
- `test_fleet_wake_retry.py:72` `test_refused_first_conversation_does_not_starve_a_ready_second`.
- `test_console_fleet_history_navigation.py:92/94` `test_view_all_pages_to_old_child_and_drills_into_it[size0]` and `[size1]`.
- `test_console_fleet_history_navigation.py:313` `test_expansion_scrolls_once…` (`scroll_y 0.0 > 0`).
- `test_console_fleet_wake_ui_freshness.py:201` `test_wake_turn_in_a_nonviewed_session_flips_the_tab_glyph_off_running`. TASK-33621.36 lists it as flaky; here it is red 2 of 2.
- `test_console_fleet_wake_wiring.py:120` `test_mount_claims_wake_marks_before_the_first_tab_sync` (order `['sync','claim',...]`).
- `test_console_headless_wake_fires.py:205` `test_a_survivor_settling_with_no_console_mounted_wakes_the_supervisor`.
- `test_console_launch_wake.py:674` `test_a_crash_killed_child_swept_to_error_wakes_nobody_at_launch`.

**5 nodes that race load:** `test_console_automatic_work_pause.py::test_pause_copy_is_painted_with_no_live_children_and_survives_view_clear`. It was red in 5 of 6 parametrizations on the PRs and 4 on dev.
- The notice has zero size after one click and one pause (`:161-163`).
- Its diagnostic at `:58-61` walks ancestors up to the App and raises `AttributeError: '_PauseHarness' object has no attribute 'region'`, which hides the real assertion.

**The cleanup that can hang.** `Tests/Chat/test_console_fleet_wake_safety.py::test_second_wake_progresses_while_first_waits_for_real_approval` hung once for 315 s in a reviewer's 10 runs, at `gather(*wake._delivery_tasks)`.
- It did not reproduce here: 10 of 10 green with `--timeout=150`, 13.4-16.9 s, load 15-26. A probe that enters the cleanup the way a failed assertion would finished it in 0.3-1.1 s in every variant.
- The await is unbounded: `await asyncio.gather(*wake._delivery_tasks, return_exceptions=True)` (`:514`, same on dev).
- A cancelled `_deliver` can keep waiting:
  - at `console_fleet_wake.py:1159-1162` (`while not claim_task.done(): await asyncio.shield(claim_task)`, which swallows the cancel);
  - at `:1235` (`await asyncio.shield(terminal)`);
  - then in its finally at `:1273`, in `drain_automatic_primary`, whose loop (`:405-409`) swallows `CancelledError` and exits only on `self._disposed` or drained capacity.
- The test's finally sets `controller._disposed`, never the coordinator's (`wake.dispose()`).
- No open task owns any of these nodes. TASK-34871 covers a different `launch_wake` node.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the 11 deterministic nodes is classified as test drift (naming the commit that moved the code) or as a product defect, and passes under its file's plain command (or under a temporary bootstrap marker until TASK-38111 lands, which the notes record); each is shown red first
- [ ] #2 Any product defect gets a test that is red before its fix
- [ ] #3 The pause-copy test waits until the notice is laid out, and its diagnostic never raises; it passes 10 of 10 whole-file runs at `-n 4`, with the load recorded
- [ ] #4 The safety test's cleanup bounds its gather and disposes the coordinator; under a temporary edit that parks a delivery, it fails within the bound and names the parked await instead of waiting out the 300 s timeout
<!-- AC:END -->
