---
id: TASK-41943
title: "Prepared legacy Stop and SubagentStop notifications may always be refused by their source check"
status: Done
assignee: []
created_date: '2026-11-19 07:30'
updated_date: '2026-11-25 15:52'
labels:
- follow-up
- agents
- hooks
- bug-suspected
priority: medium
references:
- tldw_chatbook/Agents/legacy_hook_ownership.py
- tldw_chatbook/Agents/activation.py
dependencies:
- TASK-41639
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** Read during TASK-41639 (not reproduced): a prepared legacy Stop/SubagentStop notification is captured under `worker_guard(bridge)` but checked on the notify worker against a fresh observation with no service. The `db.agent_runs` path differs between the two, so the sets may differ (not just their order) and every such notification may be refused.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test drives a real legacy Stop (and SubagentStop) notification through capture and the notify-worker check and shows whether it is refused
- [x] #2 If it is refused, capture and check observe the same service and a genuinely changed source is still refused (red first)
- [x] #3 A legacy PreToolUse hook fired under an agent admission runs and decides, instead of failing closed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (corrects which sources the existing ADR-285/148 legacy-hook custody observes; no new contract)

1. Drive real legacy Stop and SubagentStop notifications through `prepare_notification` (under `worker_guard(bridge)` and the agent service's admission, as the Console does) and `notify(_prepared=...)` with a file-backed AgentRuns database, and record where they are refused.
2. If refused: make `_EventSource` capture and check the same observation (the service-less legacy sources), so the capturing thread's admission and the notify worker's lack of one no longer differ.
3. Tests in a new `Tests/Agents/test_legacy_prepared_notification_source.py`: both notifications run their hook; a workspace database installed between prepare and delivery is still refused; red on dev, plus a negative control that drops the check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Finding: refused, twice over.** With the Console's agent bridge over a file-backed AgentRuns database, every prepared legacy Stop and SubagentStop notification was refused on dev 2959b48314. It never reached the suspected notify-worker check: `prepare_notification` already logged "future notification omitted (RuntimeError)" and returned None, so the Console never called `notify` at all.
- `_EventSource` captured `captured_sources(None)`, which under `worker_guard(bridge)` or the agent service's `execution(...)` is the caller's admitted set (`db.agent_runs` at the AgentRuns file, the ChaChaNotes path, ...).
- `capture()` then entered `execution(sources=captured)`. That also observes the service-less family sources, so it adds `("db.agent_runs", <config selector>)`.
- `admit`'s check compared the grown admitted set with the capture and refused.
- A probe that fixed only that nested admission (`resolve=` the captured tuple) moved the refusal to the suspected site. On the notify worker, which holds no admission, the check observed `_sources(None)` and refused: "notification omitted (exception_type=RuntimeError)" for both events.
- A legacy PreToolUse hook fired under the same admission (agent runs' `guard_tool_calls` via `wrap_review`) failed closed the same way. **User-visible effect on dev:** with the agent runtime on, every agent tool call matched by a legacy PreToolUse hook was blocked with "hook: hooks engine error; failing closed", and no Stop or SubagentStop hook ever ran. AC #3 was added in review (PR #3778, 2026-11-25) because the fix covers it; `CHANGELOG.md` gets an `[Unreleased] ▸ Fixed` line.

**Fix** (`Agents/legacy_hook_ownership.py`). Legacy hooks are external processes and use only the service-less agent recovery sources, so `_EventSource` now captures `selected_sources(None)`, and `check()` compares a fresh `selected_sources(None)` with it as sets. Capture and every check observe the same service, whichever admission the capturing thread holds and on any thread: the capture thread, the notify worker, the target pool and the launch check. A source that really changes between capture and check still refuses. Before, a check made inside the capture's own admission compared the admitted tuple with itself and could not see such a change. The TASK-41639 order-free comparison is kept (`set(...)`).

**Tests** (new `Tests/Agents/test_legacy_prepared_notification_source.py`, real engine, hook process and `ConsoleAgentBridge` over a file AgentRuns DB, built on `test_native_legacy_hooks.native_hooks`):
- `test_prepared_legacy_notification_runs_its_hook[Stop|SubagentStop]` prepares Stop the way `hold_legacy_stop` does (`worker_guard(bridge)`) and SubagentStop inside `execution(bridge)`, delivers it with `notify(_prepared=...)`, and waits for the hook's marker file.
- `test_prepared_notification_refuses_a_source_changed_after_capture` installs another `WorkspaceDB` registry between prepare and delivery. The notify future fails with "native hook source changed", capacity drains and no hook runs.
- `test_pre_tool_use_under_an_agent_admission_runs_its_hook` (AC #3): not blocked, and the marker exists.
- `test_pre_tool_use_refuses_a_source_changed_while_its_target_waits` (review): PreToolUse's capture-to-launch gap is a target queued in the blocking pool. The test holds all four pool workers, fires under the bridge's admission, installs another workspace database while the target is queued, then releases the workers. The target's own check raises "native hook source changed". The outcome is blocked ("hook raised; failing closed") and no hook process starts.
- `_on_thread` runs on a daemon thread and re-raises the worker's exception in the caller (review).

**Evidence.** Red on dev: all 5 failed. Three refused at prepare (custody None). PreToolUse returned "hooks engine error; failing closed". The pool test's target never queued. Negative control A (capture fix kept, `check()` ignoring sources): only the two changed-source tests failed, because the hook ran. Negative control B (dev code with only `capture()`'s nested admission fixed): both notification tests failed at the notify worker, as described above.

**Regression runs** (`-n 4`, load 30-55 on 18 CPUs). `test_native_legacy_hooks.py`, `test_run_hooks.py`, `test_hook_permissions.py`, `test_native_legacy_hook_handoff.py`, `test_console_agent_storage_sources.py`, `test_console_run_hooks_regressions.py`, `test_console_viewless_hooks.py`, `test_console_send_hook_read_sharing.py`, `test_console_local_review_hook.py`, `test_owned_hook_acceptance.py`:
- 470 passed, 5 failed, against 466 passed and the same 5 failed on dev (the 4 new tests account for the difference);
- the shared reds are 4 in `test_native_legacy_hook_handoff.py` (already reported in TASK-41639) and `test_console_run_hooks_regressions.py::test_hook_refusal_preserves_pending_attachment`.

A wider set (`test_console_prompt_mixin_admission.py`, `test_native_plugin_hook_preparation.py`, `test_console_fleet_wake.py`, `test_native_change_review_ownership.py`, `test_native_v2_mcp_events.py`, `test_fleet_send_to_agent.py`, `test_hooks_v2_lifecycle.py`):
- 233 passed, 31 failed, 14 errors;
- the same 31 nodes fail on dev source (31 failed, 13 errors). Most of `test_fleet_send_to_agent.py`, including `test_subagent_stop_fires_on_child_settle[*]`, dies in a config read with `RecoveryRequired: raw_source_selection_changed`;
- the one extra teardown error (`test_session_end_nested_workers_consume_original_holds_after_capacity_close`, a pytest-timeout after the failure) did not recur: that node passed alone on this branch.

**Final runs** (review round, after rebasing onto dev aecdc35670; load 16-37):
- The new file whole: 5 passed serially (11.0 s) and at `-n 4`.
- `Tests/CI` whole: 1946 passed, 2 skipped.
- `ruff check tldw_chatbook --select F` is clean, `Tests/lint/ruff_gate.py` exits 0, and `./scripts/preflight.sh` passed under a scratch HOME.
- The regression and wider-suite runs above are the first revision's. The review changes touch only the test file, the changelog and the backlog.

**Follow-up.** TASK-42604 covers three sibling sites with the same capture/check shape: `hooks_v2/native_commands.py`, `Chat/console_plugin_hook_preparation.py` and `Chat/console_reference_preparation.py`. It has red-first ACs.

**Not done: PR lane.** Review asked to add the new file to `.github/workflows/derived-artifacts.yml`'s "admission-sensitive suites 4" (the TASK-42281 precedent) and to `Tests/CI/test_ci_queue_pressure_contract.py::ADMISSION_4_TARGETS`. The edit was refused by this session's permission policy (no workflow edits from an agent), so under tiered CI the file runs only in this PR's light job and in full/nightly runs. Measured for that edit: 5 tests, 5-11 s serially at load 23-30. `Tests/CI` whole without the edit: 1946 passed, 2 skipped.

Files: `tldw_chatbook/Agents/legacy_hook_ownership.py`, `Tests/Agents/test_legacy_prepared_notification_source.py`, `Docs/Architecture/console-run-hooks.md` (one paragraph).
<!-- SECTION:NOTES:END -->
