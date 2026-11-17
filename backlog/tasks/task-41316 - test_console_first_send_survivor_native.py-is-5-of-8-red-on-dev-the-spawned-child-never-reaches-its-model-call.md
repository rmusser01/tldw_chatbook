---
id: TASK-41316
title: "test_console_first_send_survivor_native.py is 5 of 8 red on dev: the spawned child never reaches its model call"
status: In Progress
created_date: 2026-11-16 19:55
labels:
- follow-up
- tests
- console
- agents
priority: medium
references:
- Tests/Chat/test_console_first_send_survivor_native.py
- Tests/Chat/test_console_skill_substitution.py
- Tests/console_provider_doubles.py
- https://github.com/rmusser01/tldw_chatbook/pull/3539
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down, where runs of this file hit subprocess timeouts. TASK-40719's notes (PR #3539) say every variant "fails before reaching the save on both trees on this machine (the child never enters), so CI must qualify" its new `test_runtime_first_send_saves_through_a_wrapped_config_writer`. Re-run on dev 5e3c8b2bd4 (2026-11-16, load 33-45, scratch-profile wrapper, `-n 4`): 5 failed, 3 passed, with no timeouts this time. Line numbers are that tree's.

**Why.** This file proves that the first-send setting is saved only once a surviving sub-agent and its original sources are settled. TASK-40719's wrapped-writer fix relies on it. It runs on no PR lane (it is not in `scripts/ui_pr_gate_census.txt` or a workflow target), so nothing has qualified that fix.

**Failures.** Red nodes:
- `test_runtime_first_send_keeps_original_surviving_child`;
- `test_runtime_first_send_saves_through_a_wrapped_config_writer`;
- `test_runtime_first_send_fences_new_received_intake_during_publication`;
- `test_runtime_first_send_waits_for_late_subagent_stop`;
- `test_runtime_first_send_retains_changed_original_source`.

Each private child exits 1 at `assert child_entered.is_set(), task.result() …` (`:128`). The parent's turn completed (`ConsoleSubmitResult(accepted=True, …, visible_copy='actual parent returned', terminal_status=COMPLETED)`), but the child that the parent's `spawn_subagent` fence asked for never made its model call. The outer assertion is `:300` (`result.returncode == 0`).

Lead, not confirmed: the gateway subclasses `Tests/Chat/test_console_skill_substitution.py`'s `_RecordingGateway`. Its resolution (`Tests/console_provider_doubles.provider_resolution`) has no `continuation_protocol`. TASK-40715's notes say child runs in these hosts need one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task notes name why the spawned child never reaches its model call, from a traceback or log of the child run
- [ ] #2 All 8 nodes pass on dev at `-n 4`, including the wrapped-config-writer node; any product defect found is fixed with a test that is red first
- [ ] #3 The file runs on a PR lane, or the task notes record why it should not
<!-- AC:END -->

## Implementation Plan

ADR required: no. The private child's setup is out of date with the product; no product decision.

1. Probe the private child: dump the parent and sub-agent run steps from AgentRuns to name why the child never reaches its model call.
2. Fix the cause in the test setup or the shared double (`provider_resolution`, shared with TASK-41315), not per file.
3. Run all 8 nodes at `-n 4`; re-run any timeout alone and record load.
4. Preflight; Tests/Architecture.
