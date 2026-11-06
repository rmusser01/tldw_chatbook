---
id: TASK-39044
title: "Sixteen fleet-wake, goal-dispatch and fleet-history test nodes are red on dev for reasons other than RecoveryRequired, and the fleet-wake safety test can hang 300 s in its cleanup"
status: Done
assignee: ['@claude']
created_date: 2026-11-06 10:09
updated_date: '2026-11-06 17:26'
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
- [x] #1 Each of the 11 deterministic nodes is classified as test drift (naming the commit that moved the code) or as a product defect, and passes under its file's plain command (or under a temporary bootstrap marker until TASK-38111 lands, which the notes record); each is shown red first
- [x] #2 Any product defect gets a test that is red before its fix
- [x] #3 The pause-copy test waits until the notice is laid out, and its diagnostic never raises; it passes 10 of 10 whole-file runs at `-n 4`, with the load recorded
- [x] #4 The safety test's cleanup bounds its gather and disposes the coordinator; under a temporary edit that parks a delivery, it fails within the bound and names the parked await instead of waiting out the 300 s timeout
<!-- AC:END -->

## Implementation Plan

1. Reproduce the 11 nodes on this base under a scratch `-p` plugin that stamps `bootstrap_profile` on every item (`$FU/r11m-plug/r11m_boot.py`, no tree edit), and record each red.
2. For each node, trace the failing read to the production lines it depends on and classify it: test drift (name the commit that moved those lines, via `git log -S`/`-L`/blame, and confirm the test predates it) or product defect (consequence and mechanism). Diagnosis was split across three read-only forks of this session (wake delivery, fleet UI, goal dispatch plus the safety cleanup); each fork edited no tracked file.
3. Drift: update the test to the moved contract without weakening what it pins. Defect: a test red before the fix, then the fix.
4. Pause-copy test: wait on a bounded predicate that the notice has a laid-out region (no fixed timer), and make its diagnostic unable to raise; 10 whole-file runs at `-n 4` with the load.
5. Safety test: capture the delivery tasks, `wake.dispose()`, cancel, `asyncio.wait(..., timeout=10)`, and fail naming each parked task's await chain; show it with a temporary edit that parks `drain_automatic_primary`.

## Implementation Notes

**Approach.** The 11 deterministic nodes were reproduced on this base under a scratch `-p` plugin that stamps `bootstrap_profile` on every item (`$FU/r11m-plug/r11m_boot.py`, no tree edit): 10 failed and 1 passed in the first `-n 4` run (`[size0]` passed once and failed in later runs). Each failure was traced to the production lines it reads. Three read-only forks of this session did the tracing (wake delivery, fleet UI, goal dispatch plus the safety cleanup), and I checked each finding before changing anything. Nine nodes are test drift. Two product defects, both from 3e618d7119 (2026-10-15), account for the other two deterministic nodes and for the pause-copy race.

**Classification (AC#1).** Every node below was red on the base and passes on this branch under the scratch marker.
| Node | Kind | Cause |
|---|---|---|
| `test_console_fleet_wake_staleness.py::test_deferred_wake_notice_reports_done_despite_a_pinned_snapshot` | drift, b5251e9a6e (2026-09-16) | `connection()` became an admitted storage operation. Held open across awaits to pin a snapshot, it made the wake's config read on the loop thread refuse with `RecoveryRequired('operation_provenance_invalid')`, and `_priority_allows` failed closed. The test now takes the pinning cursor inside a short scope and keeps it open, and the "running" mechanism precondition still holds. |
| `test_console_goal_dispatch.py::test_goal_native_dispatch_is_automatic_fresh_and_returns_exact_native_outcome` | drift, 7970fbcada (2026-11-01) | A turn's mix-in attempt anchors to the last USER row and refuses a row that is not in the database (`mixin_anchor_ownership`, "Mix-in attempt could not be saved"). The test's old USER row was never saved, and is now saved (`persist=True`). |
| `test_console_headless_wake_invariants.py::test_a_headless_wake_takes_the_same_agent_dispatch_and_budget` | drift, d24c4c04b2 (2026-10-12) | `ConsoleAgentBridge.run_reply` gained `parent_budget_deadline` and `on_primary_budget_settled`. The pin names them, and also asserts that a wake passes the same deadline as a manual send. |
| `test_fleet_wake_retry.py::test_refused_first_conversation_does_not_starve_a_ready_second` | drift, 7212692c8b (2026-09-29) and b5251e9a6e | A pre-claim `pending_progress_sources` hop and admitted DB hops put the first payload 0.95-1.31 s after the drain under load, past the test's 0.5 s window. The test uses the default settle and checks fairness by submit order: the ready second never waits behind a retry of the refused first. |
| `test_console_fleet_history_navigation.py::test_view_all_pages_to_old_child_and_drills_into_it[size0]`, `[size1]` | defect A, plus drift from 3e618d7119 | Rows now land asynchronously, so the one-shot auto-open (and its reveal scroll) comes after setup. The test waits for it, waits for the closed section's toggle, and scrolls the toggle into view before clicking. |
| `test_console_fleet_history_navigation.py::test_expansion_scrolls_once_and_in_place_refresh_preserves_reading_position` | drift, ae9093a714 (2026-09-14) | Row heights moved to utility classes in the app bundle. `_ScrolledSectionHarness` loads that bundle now, which also turns `test_row_past_the_fold_is_caught_by_the_compositor_hit_test` green. |
| `test_console_fleet_wake_ui_freshness.py::test_wake_turn_in_a_nonviewed_session_flips_the_tab_glyph_off_running` | defect B | See below. |
| `test_console_fleet_wake_wiring.py::test_mount_claims_wake_marks_before_the_first_tab_sync` | drift, 3e618d7119 | The readiness projection publishes a whole sync from a worker that can run before `ChatScreen.on_mount`, so tab syncs ran before the claim in about half the mounts. The outcome the order protected still holds: receipt-gated clears (cf081bf725) and the `wake_owed` guard keep the active chat's mark. The test is renamed `test_no_tab_sync_before_the_mount_claim_consumes_the_active_chats_wake_mark`. It seeds a staged wake on the active chat before mount and asserts that every sync before the claim still saw the mark and that the wake is seeded. Negative control: with both gates temporarily removed (an unconditional view-clear), it was red 4 of 4 ("a tab sync consumed the active conversation's wake mark before the mount claim read it"). |
| `test_console_headless_wake_fires.py::test_a_survivor_settling_with_no_console_mounted_wakes_the_supervisor` | drift, b7dd8e53f2 (2026-09-08) and fed0b7257a (2026-09-05) | Leaving Console now only detaches its view and no longer sets `_shutdown_requested`, so the precondition is inverted, with the reason given in the test. The chat route reuses its screen, so `_navigate("chat")` brought back a second, superseded view in this harness. The return now mounts a fresh `ChatScreen`, as `_seed_console` does. |
| `test_console_launch_wake.py::test_a_crash_killed_child_swept_to_error_wakes_nobody_at_launch` | drift, 827b622638 (2026-09-11) | Launch discovery reads the ledger, not the marks ("a missing badge never hides a pending result"), so the orphan builds the runtime. The test asserts what keeps it silent instead: its chainless result is paused for review (`legacy_lineage`). The docstring now says so too. |

**Product defects (AC#2), both from 3e618d7119.**
- **A. A config write reopened a collapsed Agents fleet and scrolled the rail.** `_presentation_historical_snapshot` keys its cached read on the config identity, and on a re-key it served empty rows until the new read landed. Empty rows reset `_console_fleet_section_auto_opened` and `_console_fleet_inspector_dismissed`, so the section the user had collapsed auto-opened again and `_reveal_expanded_section` scrolled the rail. The fix: a re-key for the same owner and chat (key fields 0-2 and 4-8 equal; only the config identity or the live status moved) keeps serving the last rows until the new read lands, and a new read that comes back empty still repaints. New test `test_console_fleet_history_navigation.py::test_a_config_write_keeps_a_collapsed_fleet_collapsed_with_its_rows` (the config write is a real Settings save of `tool_result_display_chars`). Against the base `agent.py` it was red 3 of 3 ("a config write re-opened the fleet section the user collapsed"); with the fix, green 3 of 3.
- **B. A wake turn that finished in a background chat never got its ✓, toast or receipt.** `ConsolePromptQueueCoordinator.activity()` counts a live received claim as occupying the slot. A chainless automatic turn (agent wake, agent chat start, goal iteration) keeps its own claim until `ConsoleRuntime._release_custody`, which runs after `_set_run_state(COMPLETED)`. At its one terminal transition the turn was therefore never `terminal_notification_eligible`, and the tab fell back to ◈. The fix: an automatic-origin claim held at a terminal status does not withhold eligibility. Slot occupancy is unchanged, and so are manual and queued claims and chain turns, which publish at chain end. The red-before test is the node itself (red on the base in every run, green on this branch).
- The User Guide already documents both behaviours (`Docs/User_Guide/console/agent-runs-and-tools.md`: a tab that finishes unseen flips to ✓), so no page changed.

**Pause-copy test (AC#3).** It closed the section before the asynchronous auto-open (defect A plus drift), read the notice after a single pause, and its diagnostic walked up to the App, which has no `region`. It now waits (bounded) for the auto-open, except for `history_unavailable`, which has no rows; then for the closed section's recomposed toggle; then for the notice's region to have area. Its diagnostic skips the App and reads `region` with `getattr`. Result: 10 of 10 whole-file runs at `-n 4` (13 passed each, load 24-58), plus 9 of 9 with three copies of the file running at once (load 35-40).

**Safety cleanup (AC#4).** The `finally` keeps the delivery tasks, calls `wake.dispose()` (which ends `drain_automatic_primary`'s wait but empties the registry), cancels them, waits at most 10 s, and fails naming each parked task's coroutine chain (`_awaiting`). Plain runs: 3 of 3 (19-21 s). Negative control, a temporary edit that parks `drain_automatic_primary` for 20 s ignoring dispose and cancel: it failed after the bound, 34.8 s in all instead of 300 s, with "wake delivery still parked after 10 s: ConsoleFleetWakeCoordinator._deliver:1274 -> ConsoleFleetWakeCoordinator.drain_automatic_primary:408 -> sleep:665". The fork showed the reported hang with a child-only probe: with the drain kept busy, the old cleanup sat until the probe's 45 s limit, and the new one ended at once.

**Runs.** The 12 files (the task's 11, plus `test_console_inspector_section.py`), `-n 4`, under the scratch marker, load about 42: 123 passed, 2 failed. The base under the same command gave 17 failed and 107 passed. Both remaining failures (`test_summary_is_right_aligned_within_the_header_region_no_chevron`, `test_summary_sits_between_title_and_chevron_when_collapsible`) fail the same way on the base, in a different harness, and are outside this task. Plain (without the marker), the UI and fleet-wake Chat files still stop on TASK-38111's `RecoveryRequired` ("Hooks unavailable" for the Chat rigs); the marker stays until TASK-38111 lands. Eligibility covering set, 21 files: 9 failed and 509 passed on this branch, against 14 failed and 504 passed on the base, and every failure on this branch also failed on the base. Agent-rail covering set, 14 files: 14 failed and 289 passed, against 16 failed and 287 passed on the base. The one failure only this branch showed (`test_console_parallel_runs.py::test_background_skill_script_confirm_parks_badges_toasts_and_mounts_on_visit`, at load about 50) passed 2 of 2 alone, and passed on the base alone. Load-only flakes seen on this branch: `test_history_remains_reachable_after_a_childless_primary_run` (once in a 12-file run; 3 of 3 file runs and 2 of 2 alone green); goal-dispatch neighbours at load 35 (2 of 9 runs; 5 of 5 later); the freshness node at :174, "Mix-in attempt could not be saved" (2 of 6 at load 58).

**Not laned.** No PR lane runs these files, and the criteria ask for none. They need TASK-38111 first.

**Files.** `tldw_chatbook/UI/Console_Modules/agent.py`, `tldw_chatbook/Chat/console_prompt_queue_coordinator.py`; tests in `Tests/Chat/` (`test_console_fleet_wake_staleness.py`, `test_console_goal_dispatch.py`, `test_console_headless_wake_invariants.py`, `test_fleet_wake_retry.py`, `test_console_fleet_wake_safety.py`) and `Tests/UI/` (`test_console_fleet_history_navigation.py`, `test_console_inspector_section.py`, `test_console_fleet_wake_wiring.py`, `test_console_headless_wake_fires.py`, `test_console_launch_wake.py`, `test_console_automatic_work_pause.py`).
