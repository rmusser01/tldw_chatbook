---
id: TASK-41313
title: "Five run-log, fleet-budget and turn-activity test files are red on dev, mostly behind the raw_source_selection_changed config-sandbox trap"
status: To Do
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
- [ ] #1 Each of the five files passes on dev under the plain one-file command at `-n 4`, without binding config to the developer's real profile
- [ ] #2 The two activity-line repaint nodes are classified as test drift (naming the commit) or a product defect (fixed, with the test red first)
- [ ] #3 Each file runs on a PR lane, or the task notes record why it should not
<!-- AC:END -->
