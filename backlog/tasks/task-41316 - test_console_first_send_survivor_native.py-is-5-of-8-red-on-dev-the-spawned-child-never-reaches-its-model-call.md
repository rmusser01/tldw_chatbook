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
- [x] #1 The task notes name why the spawned child never reaches its model call, from a traceback or log of the child run
- [ ] #2 All 8 nodes pass on dev at `-n 4`, including the wrapped-config-writer node; any product defect found is fixed with a test that is red first
- [ ] #3 The file runs on a PR lane, or the task notes record why it should not
<!-- AC:END -->

## Implementation Plan

ADR required: no. The private child's setup is out of date with the product; no product decision.

1. Probe the private child: dump the parent and sub-agent run steps from AgentRuns to name why the child never reaches its model call.
2. Fix the cause in the test setup or the shared double (`provider_resolution`, shared with TASK-41315), not per file.
3. Run all 8 nodes at `-n 4`; re-run any timeout alone and record load.
4. Preflight; Tests/Architecture.

## Implementation Notes

Outcome: 7 of 8 nodes pass; `test_runtime_first_send_retains_changed_original_source` is a product defect, not fixed here (stopped for a decision).

**Cause (AC #1), from the private child's own AgentRuns rows (`list_runs`/`get_run`, written to a scratch file from inside the script).** Two test-side causes, one hiding the other; neither is new:
- The parent ran on plain chat's light tool surface: its run's allowed tools were `('calculator', 'get_current_datetime')`, and the `spawn_subagent` call's `tool_result` was `ERROR: Tool not permitted: spawn_subagent`, so no child ever existed. The script's config has no `[tools] full_agent_tools_enabled`, which defaults off (TASK-34100.5.3, cf3ab5e2c8, merged as #3295 on 10-31, before this file was written).
- With Full agent tools on, the child's run row ends `error` before its model call: `unexpected provider error ('types.SimpleNamespace' object has no attribute 'continuation_protocol')`. The sub-agent continuation target in `console_streaming_request._consume_body` (bf1886c95c, 10-28) reads `effective_resolution.continuation_protocol`. The shared double `Tests/console_provider_doubles.provider_resolution` did not carry the real resolution's `continuation_protocol`/`api_mode` (both `None` by default on `ConsoleProviderResolution`).
- Both causes are older than the file. It has been red since it reached dev: red at #3480's merge (e10ce63fe8) and at that branch's tip (fe08463ab1). Earlier commits on that branch fail at import (`config.py` NameError for `get_canvas_execution_enabled`), so no green ancestor exists to name.

**Changes.**
- `Tests/console_provider_doubles.py`: `provider_resolution` defaults `continuation_protocol` and `api_mode` to `None`, like the real dataclass. This is one fix for all 56 users of the double. It also covers TASK-41315's file and makes the widening test's explicit override redundant (left in place).
- `Tests/Chat/test_console_first_send_survivor_native.py`: the script's config turns on `[tools] full_agent_tools_enabled`. The hang detectors are widened (child entry 12 s to 60 s, journey 35 s to 120 s, subprocess 45/50 s to 150 s; see TASK-40306's lesson). At load 75-90 the journey takes 40-70 s, and the `admission` variant (which runs `setprofile_all_threads`) took 66 s alone.

**Evidence.** Each cause was confirmed by fixing only the other: with only the double fixed, the child is still `Tool not permitted`; with only the config fixed, the child still fails on `continuation_protocol`. With both fixed, the `native` node passes alone (28 s at load 56). Final run at `-n 4` (load 55): 7 passed, and only `retains_changed_original_source` failed. An earlier run at load 83-115 also failed `admission`, on a 30 s child-entry detector now widened to 60 s; that node passes alone in 66 s. `writer_wrapped`, which qualifies TASK-40719's fix, passes.

**Not fixed: product defect (source_loss).** After the script swaps `TLDW_CONFIG_PATH` (so `config.current_config_identity()` changes) and releases the surviving child, the child completes. Its row is stored `subagent done` with its reply, nothing raises `original agent storage source changed`, and the tree has not retired. The result is the same with TASK-41315's product change reverted. The child path no longer re-checks `_AgentSources._check_original_sources` after its model call returns. Deciding where a surviving child must re-check its sources touches child custody, so this needs a decision.

**AC #3.** Not ticked: workflows are off-limits to this task. The file should join `pr-fast-lane-admission` ("Run admission-sensitive suites") once the source_loss node is resolved, or join now with that node deselected.

Files: `Tests/console_provider_doubles.py`, `Tests/Chat/test_console_first_send_survivor_native.py`.
