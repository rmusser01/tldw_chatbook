---
id: TASK-41313
title: "Five run-log, fleet-budget and turn-activity test files are red on dev, mostly behind the raw_source_selection_changed config-sandbox trap"
status: In Progress
created_date: 2026-11-16 19:55
labels:
- follow-up
- tests
- agents
- console
priority: medium
references:
- Tests/Agents/test_run_log_prompt_integration.py
- Tests/Agents/test_run_log_stats_slice_runtime_tools.py
- Tests/Agents/test_run_log_survivor_lifetime.py
- Tests/Chat/test_fleet_budget_boundary_probes.py
- Tests/UI/test_console_turn_activity_line.py
- Tests/conftest.py
- tldw_chatbook/Agents/tool_catalog.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. TASK-40715's PR #3548 listed the run-log suites as "pre-existing ... red on dev". The other two files were red in the same burn-down's base runs. Re-run on dev 5e3c8b2bd4 (2026-11-16, load 33-45, scratch-profile wrapper, `-n 4`); line numbers are that tree's.

**Why.** These files are red on dev and run on no PR lane: none is in `scripts/ui_pr_gate_census.txt` or a workflow target. A regression in run-log survivors, fleet budgets or the Console activity line would land unseen. Most of the reds share one cause. None of the five files has a `bootstrap_profile` marker, so the per-test config sandbox makes config admission refuse with `RecoveryRequired('raw_source_selection_changed')` (`Backup_Recovery/raw_participants.py:307`). That is the trap TASK-38111 fixed for other files.

**Runs.**
- Run-log suites (three files): 8 failed, 8 errors, 144 passed.
  - `BuiltinToolProvider()` (`Agents/tool_catalog.py:1451`) reads each gate through `get_cli_setting`, gets RR, and logs "Could not register builtin tool …". The fleet children then have no tools and end `error`. The tests fail on that: `assert 'error' == 'done'` (`test_run_log_survivor_lifetime.py:164`, `:234`, `:498`; `test_run_log_stats_slice_runtime_tools.py:226`), "expected exactly one sub-agent row, got 0" (`:132`), and `assert [] == ['running']` (`:443`).
  - `test_run_log_prompt_integration.py:192`: "expected at least one call_model turn".
  - The 8 errors are the conftest's teardown check: "FleetChat scripting fault(s): parent has 2 unused turn(s); child … has 1 unused turn(s)".
- `Tests/Chat/test_fleet_budget_boundary_probes.py`: 5 failed, 1 error (`assert 0 == 1` at `:130`, `assert False` at `:166` and `:228`, an unset `threading.Event`). The log carries the same RR from builtin-tool registration and the same FleetChat teardown fault.
- `Tests/UI/test_console_turn_activity_line.py`: 21 failed, 30 passed.
  - 18 hit RR when the mounted transcript reads `console.turn_file_cards` through `get_cli_setting` (`Widgets/Console/console_transcript.py:4777`).
  - `test_a_held_tool_call_names_the_tool_in_the_rendered_assistant_row` fails with "the scripted tool call never reached the tool" (`:238`): its builtin tools failed to register with the same RR.
  - Two are not RR: `test_a_ticking_elapsed_alone_repaints_the_transcript` (`:1059`) and `test_published_token_usage_repaints_once_per_changed_scalar` (`:1076`) see `refresh_messages.await_count` 1, not 2. `ChatScreen._sync_native_console_transcript` no longer repaints when only the activity line changes. That may be test drift or a lost repaint.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the five files passes on dev under the plain one-file command at `-n 4`, without binding config to the developer's real profile
- [x] #2 The two activity-line repaint nodes are classified as test drift (naming the commit) or a product defect (fixed, with the test red first)
- [ ] #3 Each file runs on a PR lane, or the task notes record why it should not
<!-- AC:END -->

## Implementation Plan

ADR required: no. This applies the existing `bootstrap_profile` test marker (TASK-38111 / TASK-32873) and classifies two test nodes; no product boundary changes.

1. Reproduce all five files on base (fu-base, `-n 4`) and confirm the `raw_source_selection_changed` refusal in each file's own tracebacks.
2. Add `pytestmark = pytest.mark.bootstrap_profile` to each file; re-run each file alone at `-n 4` under the scratch-profile wrapper.
3. For the two repaint nodes, find the commit that changed `ChatScreen._sync_native_console_transcript` (git log -S/-L) and classify as test drift (update the test, name the commit) or a lost repaint (red test, then fix).
4. Fix any residual non-RR reds the marker exposes; report the lane each file should join (no workflow edits in this PR).

## Implementation Notes

**Lead confirmed, then partly wrong.** On base (fbf1c9ea74) the five files gave 34 failed, 9 errors, 174 passed, with `raw_source_selection_changed` in every red file's log. A module `pytestmark = pytest.mark.bootstrap_profile` (the TASK-38111 / TASK-32873 marker) made the three run-log files green. Five non-RR reds were left:

- **Fleet budget probes, test drift (3 nodes).**
  - `test_terminal_status_cannot_release_runtime_capacity_before_worker_exit` patched `FLEET_JOIN_TIMEOUT_SECONDS` to 0.01 s. Since 06f8b0bf94 (TASK-39147) that constant also bounds the child's admission handshake (`admission.result(timeout=...)`, now in `agent_native_execution.py`). The child was refused before its call: `AgentActivationRequired` at `run_child`, which showed up only after a temporary `logger.opt(exception=True)`. The test now uses 1.0 s, under its 5 s release wait.
  - `test_timed_out_tool_workers_cannot_accumulate_within_one_run`: its 0.02 s timeout can fire before a loaded worker reaches the tool, so `workers` was empty. The test now waits for the tool to be entered before it counts.
  - `test_successive_wakes_stop_at_shared_generation_limit` took 5.6 s at load 30 against a 5 s `_settle`. It is slow, not wedged (it passes serially in 3.6 s). It now has 30 s.
- **Activity-line repaints: product defect (2 nodes), fixed.** The task-34920 gate (c6d91555a4) skips the transcript pipeline on an unchanged gate tuple. 1670513778 (review I1) added a run-activity component "so turn-activity clocks ... stay live", but that component (run state, approval flag, round kinds) never carries the elapsed seconds, the token figures or the abandon affordance. So during a store-quiet tool call the line froze until the next forced pass, about 3 s. `_sync_native_console_transcript` now reads the line and its abandon action once per tick, before the gate, and puts both into the gate tuple; the same values go to `apply_turn_activity`. The two nodes were red on base and red with only the marker (`assert 1 == 2`), and they pass with the fix.
  - Known cost: during an active run the pipeline now runs at most once a second while the line changes, against once every 15 ticks before. The line is read on every tick. In the first-token wait (more than 15 s with an idle bridge snapshot), that read includes one `read_only_messages_for_session` snapshot.

**AC #3 (left open: workflow edits are out of scope for this PR).**
- The four non-UI files are `bootstrap_profile` suites. They belong in `recovered-suites` ("PR lane (recovered suites)", `.github/workflows/derived-artifacts.yml`), in the bootstrap block after the sandboxed suites.
- `Tests/UI/test_console_turn_activity_line.py` belongs on the UI Fast Lane, through a tail entry in `scripts/ui_pr_gate_census.txt` (charge and rebalance per its header).

**Verification** (2026-11-16/17, base fbf1c9ea74, load 30-150, scratch-profile wrapper):
- Each of the five files passes alone at `-n 4`: 10, 134, 8, 5 and 51 passed.
- Mutation: with only the `chat_screen.py` hunk reverted, both repaint nodes fail again (`1 == 2`). Restoring it makes them pass.
- Base vs head, one file per run at `-n 4`, over the 24 files that call `_sync_native_console_transcript`: the failing node sets are identical. The one exception is `test_accepted_agent_chat_start_has_visible_stop_in_mounted_target`, which failed once at load 100 and then passed 3 of 3 runs alone.
- Already red on base, from the same RR (no marker), and not touched here: `test_console_native_transcript` (110 failing nodes), `test_change_turn_tracking` (46), `test_console_composer_collapse` (30), `test_console_annotation_markers` (14), `test_console_transcript_context_projection` (9), `test_console_selection_end_to_end` (5), `test_settings_console_thinking` (5), `test_console_turn_attention` (3), `test_console_poll_reconciliation` (2), `test_console_undo_restore_pacing` (1).
- `Tests/Architecture`, serial: the first version of the fix put `chat_screen.py` 2 lines over its size ratchet. Trimmed, it is at 24565 of 24569. The remaining 3 failures (`test_timer_path_static_update_inventory` x2, `test_reviewed_diagnostic_changes_are_metadata_only`) also fail on base.
- Serial times, for the lane decision: the four non-UI files take 131 s in one invocation, the UI file 42 s.

Files: the five test files (module marker), `Tests/Chat/test_fleet_budget_boundary_probes.py` (three drift fixes), `tldw_chatbook/UI/Screens/chat_screen.py` (gate component).
