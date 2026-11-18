---
id: TASK-38978
title: "The real-key Ctrl+Q in-flight modal tests run on no PR lane"
status: Done
assignee: ['@claude']
created_date: 2026-11-06 01:42
updated_date: '2026-11-07 02:19'
labels:
- follow-up
- ci
- tests
- quit
priority: low
references:
- Tests/UI/test_app_quit_in_flight_modals.py
- scripts/ui_pr_gate_census.txt
- scripts/check_ui_pr_gate_census.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-38747
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b and on the census and workflows as they stand once the round-8 branches merge, 2026-11-05.

`Tests/UI/test_app_quit_in_flight_modals.py` holds the real-key Ctrl+Q tests for guarded modals (TASK-33622.15, TASK-38747). It is in no workflow and not in the census.

The census checker's comment (`scripts/check_ui_pr_gate_census.py`, about lines 100-104 after the merges) keeps the file out, saying it takes "~2.7 min" and stays off "until the lane has a shard with room". The measured time is far lower:

| Run | Result | Time | Load |
|---|---|---|---|
| Serial at `--timeout=180` (the UI Fast Lane configuration) | 13 passed | 47.8 s | 5.8 |
| `-n 4`, 3 runs | 13 of 13 each time | 25-29 s each | 11.9-12.5 |

TASK-38747 adds 3 tests to the file, making 16.

TASK-33622.15 and TASK-33622.17 (both Done) only noted that the file was off the lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The file runs in the UI Fast Lane, and the census checker's comment no longer says it stays out
- [x] #2 Before it is added, it passes 3 of 3 serial runs at --timeout=180 on dev with TASK-38747's tests, with the measured time recorded (47.8 s for 13 tests on dev)
- [x] #3 The shard it joins still ends at least 5 minutes under its 20-minute cap in its slowest measured run
<!-- AC:END -->

## Implementation Plan

1. Qualify before adding: `Tests/UI/test_app_quit_in_flight_modals.py` 3 serial runs at `--timeout=180` (the UI Fast Lane's flags) on dev (the read-only base worktree, 16 tests) and on this branch (17 with TASK-38980's promote test), with times and load.
2. Read recent UI Fast Lane shard times (`gh run view --json jobs`, read-only) and pick the tail slot by the round-robin deal (`entries[index::32]`); record the slowest run of the shard it joins plus the file's measured time, which must stay at least 5 minutes under the 20-minute cap.
3. Append the file as one whole-file entry at the census tail, set `MINIMUM_FILES` to the new count with a dated comment, and rewrite the checker's "stay out until the lane has a shard with room" comment to say where the file runs now.

## Implementation Notes

**Approach.** `Tests/UI/test_app_quit_in_flight_modals.py` joins the UI census at the tail as one entry per test (see the shard note), `MINIMUM_FILES` goes to the new entry count (1796 on this base, with TASK-38979's and TASK-38980's entries appended beside it), and the checker's TASK-33622.15 comment now says the file runs in the lane (`test_console_video_picker_cancel.py` is still out). No workflow file changed: the UI Fast Lane reads the census.

**Qualification (AC#2), `pytest -p no:xdist --timeout=180`, serial.**
- Base, the read-only fu-base-r11 worktree (16 tests, with TASK-38747's three): 3 of 3, 115.2 s, 125.0 s and 118.6 s at load averages 16.0-30.2.
- This branch (17 tests, with TASK-38980's Promote test): 3 of 3, 135.0 s, 118.4 s and 126.6 s at load 26.9-47.5.
- The filing measured 47.8 s for 13 tests at load 5.8. The machine was shared with other agents this round, and local wall time scales with load.

**Shard (AC#3), after review r11m-rev1.** The first cut was one whole-file entry, which put about a minute on shard 21. Its slowest recent run is 13.5 min, so the margin under 15 min rested on an estimate of the file's CI time: the reviewer's calibration put it at 1.3-3.5 min, not 1.1. The file is now 17 entries, one per test, appended at the tail. They are dealt round-robin over consecutive shards, so no shard takes more than one or two of them, and the restack can move them without loading one shard. Per-entry serial times with `--timeout=180`, measured at load about 70 (a few times what a quiet machine shows): test_ctrl_q_waits_for_a_fork_that_is_committing 19.3 s; test_ctrl_q_waits_for_a_capture_policy_change_being_applied 18.7 s; test_ctrl_q_waits_for_a_trace_privacy_change_being_applied 24.8 s; test_ctrl_q_waits_for_a_project_skills_import 19.7 s; test_ctrl_q_waits_for_a_message_delete_and_its_undo 18.8 s; test_a_stuck_operation_still_lets_a_repeated_ctrl_q_quit 19.5 s; test_a_fork_that_finishes_under_the_quit_question_closes_after_wait 17.1 s; test_ctrl_q_over_the_video_save_picker_asks_before_discarding_it 21.1 s; test_ctrl_q_over_the_video_replace_confirmation_asks_first 23.1 s; test_ctrl_q_while_the_interview_discard_runs_waits_then_quits 19.7 s; test_a_failed_interview_discard_lets_ctrl_q_ask_again 16.6 s; test_an_interview_discard_finishing_while_covered_still_closes_it 18.4 s; test_an_interview_discard_finishing_under_the_quit_question_closes_after_wait 14.8 s; test_ctrl_q_waits_for_a_goal_that_is_starting 13.5 s; test_ctrl_q_waits_for_portrait_regions_being_saved 17.9 s; test_ctrl_q_waits_for_a_rule_test_then_asks_before_discarding_edits 22.2 s; test_ctrl_q_waits_for_a_confirmed_rule_promote_being_written 20.2 s. The CI time of these entries is still estimated, not measured: no CI run has executed them yet. One census file measured both ways ran 12.1 s in a CI shard log and 10.5 s here at load 31. Every entry collects: the 48 TASK-38978, TASK-38979 and TASK-38980 entries collect 51 tests. On the base before the split, whole shards 21 and 22 passed locally (223 passed in 892 s and 169 passed in 872 s, at load 32).

**Placement (review r11m-rev2).** The tail block (the Promote node, the 17 quit-modal entries and the 30 region-editor entries) is ordered by `$FU/census_reorder.py`. Hot shards are the ones whose slowest UI Fast Lane job over the last 12 PR runs is not at least 5 min under the 20-min cap: 3 (16.9), 5 (16.7), 8 (15.3), 13 (16.1), 15 (15.0) and 19 (15.0), from `$FU/ui_shard_ci_minutes.json`. The lightest entries take the hot slots, hottest shard first, and the heaviest entries go to the coolest shards. Dealing is the checker's own (`entries[index::32]`), as `$FU/census_shards.py` reproduces. **Shortfall:** the block has 7 hot slots but only 3 entries under 5 s (at load ~70), so 4 hot slots still hold heavier entries. These are shard 8, the Promote node (13.0 s at load ~31); shard 13, `test_retiring_the_editor_before_it_mounts_still_closes_it[unmount]` (8.4 s); shard 15, `test_ctrl_q_waits_for_a_goal_that_is_starting` (13.5 s); and shard 19, `test_save_completion_drops_removed_validation_control` (14.5 s). The owner accepted these four as a stated exception (decision below). **The final placement is set by the coordinator after the round-11 restack** (re-run `census_reorder.py` on the restacked tail); the shards below are this base's.

| Entry | Shard on this base (slowest recent CI min) | Serial s (load ~70) |
|---|---|---|
| `test_ctrl_q_waits_for_a_capture_policy_change_being_applied` | 21 (13.9) | 18.7 |
| `test_ctrl_q_waits_for_a_project_skills_import` | 24 (13.9) | 19.7 |
| `test_a_failed_interview_discard_lets_ctrl_q_ask_again` | 28 (14.5) | 16.6 |
| `test_ctrl_q_waits_for_a_confirmed_rule_promote_being_written` | 29 (13.8) | 20.2 |
| `test_ctrl_q_over_the_video_save_picker_asks_before_discarding_it` | 30 (13.6) | 21.1 |
| `test_ctrl_q_waits_for_portrait_regions_being_saved` | 4 (14.1) | 17.9 |
| `test_ctrl_q_waits_for_a_fork_that_is_committing` | 10 (13.9) | 19.3 |
| `test_a_stuck_operation_still_lets_a_repeated_ctrl_q_quit` | 11 (13.9) | 19.5 |
| `test_ctrl_q_waits_for_a_message_delete_and_its_undo` | 14 (13.9) | 18.8 |
| `test_ctrl_q_waits_for_a_goal_that_is_starting` | 15 (15.0) | 13.5 |
| `test_ctrl_q_waits_for_a_rule_test_then_asks_before_discarding_edits` | 17 (13.3) | 22.2 |
| `test_an_interview_discard_finishing_while_covered_still_closes_it` | 21 (13.9) | 18.4 |
| `test_ctrl_q_over_the_video_replace_confirmation_asks_first` | 22 (13.1) | 23.1 |
| `test_a_fork_that_finishes_under_the_quit_question_closes_after_wait` | 23 (14.2) | 17.1 |
| `test_ctrl_q_while_the_interview_discard_runs_waits_then_quits` | 24 (13.9) | 19.7 |
| `test_ctrl_q_waits_for_a_trace_privacy_change_being_applied` | 25 (12.6) | 24.8 |
| `test_an_interview_discard_finishing_under_the_quit_question_closes_after_wait` | 26 (14.5) | 14.8 |

**AC#3.** **Owner decision (2026-11-06):** "Accept, record exception: lane all 47; those four shards gain a few seconds each and stay under 17 min; record the AC as met with a stated exception and file a follow-up to rebalance or split the hot shards themselves."
The criterion is met with this stated exception: four hot slots (shards whose slowest recent CI run is not at least 5 min under the 20-min cap) still hold an entry over 5 s:
  - shard 8: the Promote node, `Tests/UI/test_response_rules_modal.py::test_a_confirmed_promote_keeps_the_manager_busy_and_open_until_it_reports`, 13.0 s (serial, load ~31);
  - shard 13: `Tests/UI/test_portrait_region_editor.py::test_retiring_the_editor_before_it_mounts_still_closes_it[unmount]`, 8.4 s (serial, load ~70);
  - shard 15: `Tests/UI/test_app_quit_in_flight_modals.py::test_ctrl_q_waits_for_a_goal_that_is_starting`, 13.5 s (serial, load ~70);
  - shard 19: `Tests/UI/test_portrait_region_editor.py::test_save_completion_drops_removed_validation_control`, 14.5 s (serial, load ~70).
The coordinator is filing the follow-up to rebalance or split the hot shards.

**Verification after review r11m-rev1** (2026-11-07 UTC; the machine was shared, and the load is given for each run).
- **Head.** 19 modal and fleet files under the scratch boot plugin, `-n 4`, load 20 to 80: 272 passed, 2 failed. The two failures are the inspector-section summary tests, which also fail on base.
- **Base.** The same files: 234 passed and 17 failed, and then the run hung with idle workers. The watchdog killed it after 30 minutes, with about 20 tests unfinished. An earlier base attempt hung for more than 2 hours. Base still has the hangs these tasks fix.
- `Tests/CI -n 4`: 1215 passed, 1 skipped.
- Preflight passes.
- Architecture and size ratchets: 216 passed.
- Boot budget: 32 passed.
- **Stress.** The new and changed nodes are 22 selectors, 45 tests, run at `-n 6` under the plugin, 3 runs per round:
  - Round 1, load 65 to 104: 1 failed, 0 failed, 1 failed.
  - Round 2, load 53 to 77: 2 failed, 1 failed, 0 failed.
  - Each test-wait failure was fixed (bounded waits, a held-save precondition) or traced to the product re-sync above.
  - Round 3, on the final head, load 42 to 61: 1 failed, 0 failed, 0 failed. The one failure is the freshness node's known "Mix-in attempt could not be saved" race, an automatic turn's mix-in anchor, which is reported as a follow-up.

**Verification after review r11m-rev2** (2026-11-07 UTC, load about 25-50).
- **Head.** 19 modal and fleet files under the scratch boot plugin, `-n 4`: 272 passed, 2 failed. The two failures are the inspector-section summary tests, which also fail on base.
- **Receipt and read-state coverage.** `test_console_chat_controller.py`, `test_console_activity_receipts.py`, `test_console_receipt_bootstrap.py`, `test_buddy_hidden_results.py`, `test_console_refresh_read_batching.py` and `test_console_historical_read_cancellation_lifetime.py`: 36 failed and 376 passed on both head and base, with identical failure sets.
- `Tests/CI -n 4`: 1215 passed, 1 skipped.
- Preflight passes.
- Architecture and size ratchets: 216 passed.
- Boot budget: 32 passed.
- **Stress.** The new and changed nodes are 22 selectors, 45 tests, run at `-n 6`:
  - First round: 1 failed, 0 failed, 1 failed. Both failures were the view-all test at 120x35, now fixed: it waits for the toggle, tail and steps to be painted.
  - Final round: 45 passed in each of the 3 runs.
- **Tooling, all in `$FU`.** `collect_ui_shard_minutes.py` writes `ui_shard_ci_minutes.json`. `census_shards.py` shows the shard dealt to each census entry and flags heavy entries on hot shards. `census_reorder.py` permutes one block. The weights are in `modals_census_weights.json`.

**Files.** `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.

**Final placement on the round-11 stacked tree (coordinator, 2026-11-06).** After the restack the block starts at census entry 1762 (floor 1809). The reorder put the lightest entries on the hot slots; seven hot slots still hold an entry of 13-15 s (serial, at load ~70), on the same four shards the owner's exception names, two entries each: shard 8 (`held_save[source]` 14.6 s, `save_completion_drops_removed_validation_control` 14.5 s), shard 13 (the Promote node 13.0 s, `ctrl_q_waits_for_a_goal_that_is_starting` 13.5 s), shard 15 (`held_save[settings]` 14.9 s, `interview_discard_finishing_under_the_quit_question` 14.8 s) and shard 19 (`held_save[cancel]` 14.9 s). On CI these are a few seconds each, inside the owner's "Accept, record exception" ruling; TASK-39144 rebalances the hot shards.
