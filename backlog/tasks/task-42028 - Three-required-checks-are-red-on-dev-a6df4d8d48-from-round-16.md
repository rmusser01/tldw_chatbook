---
id: TASK-42028
title: "Three required checks are red on dev a6df4d8d48 from round 16"
status: Done
assignee:
- '@claude'
created_date: 2026-11-20 01:20
updated_date: 2026-11-20 02:07
labels:
- follow-up
- ci
priority: high
references:
- .github/workflows/derived-artifacts.yml
- Tests/Chat/test_console_agent_call_thinking.py
- Tests/UI/test_console_first_send_import_cycles.py
- Tests/UI/test_console_launch_wake.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3637's required run (head 00d2c77e11, run 44626839587) is the first full run on dev after both round-16 batches. It failed three required checks, and each failure traces to a round-16 change:

- **Console P0 regression gate (admission-sensitive)**, job 135948664292. All 9 tests in `Tests/Chat/test_console_agent_call_thinking.py` fail with `ModuleNotFoundError: No module named 'hypothesis'`. TASK-41508 moved the file into that step, and the job does not install hypothesis.
- **UI Fast Lane (31)**, job 135948665510, `Tests/UI/test_console_first_send_import_cycles.py`. Six teardown ERRORs: `import_module() takes 1 positional argument but 5 were given`. The tests patch `builtins.__import__` with a one-argument fake, and an autouse teardown imports after the test body, while the patch is still in place.
- **The same job**, `Tests/UI/test_console_launch_wake.py`. 2-4 tests per run end in teardown ERRORs with two sub-exceptions: `SqliteRetirementIncomplete: sqlite_retirement_incomplete:AgentRuns` from `dispose_console_runtime`, and `RuntimeError: Native app workers remain uncertain; config retirement refused`. The launch-wake work came from TASK-41507 (3cf6799ff0, 4f4077dbe6).

The second and third errors also reproduce locally at bare dev a6df4d8d48.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The admission-sensitive Console P0 job runs every test in test_console_agent_call_thinking.py with no import error
- [x] #2 test_console_first_send_import_cycles.py passes with no teardown error when it shares an xdist worker with test_console_fleet_wake_wiring.py, and its fake import cannot leak past the test body
- [x] #3 test_console_launch_wake.py has no teardown error in repeated -n 4 runs; the AgentRuns database a launch wake opens is retired before teardown, with the root cause named in the notes
<!-- AC:END -->

## Implementation Plan

1. AC#1: add `hypothesis` to the Console P0 gate's install line (the thinking file imports `Tests/Agents/test_fleet_continuation.py`, which imports hypothesis), and extend the CI contract test that pins that line. Check the other P0 targets for packages the job lacks.
2. AC#2: confirm the fixture order (autouse `drain_test_app_user_data_dirs` tears down before monkeypatch) and reproduce the leak with the fleet wake wiring file in the same worker; scope each fake `__import__` to the call under test so nothing after the body sees it.
3. AC#3: reproduce the launch-wake teardown errors at -n 4, find who holds the AgentRuns connection or a native worker at dispose time, and fix it at the source (product if a real wake leaks it). Show red, then green over at least 3 -n 4 runs with the load.
4. Run the touched files at -n 4, Tests/CI at -n 4, preflight, ruff F on changed files, the ruff gate and any size ratchet for touched product modules.

## Implementation Notes

**AC#1 (hypothesis).** The thinking file's resumed-child test imports helpers from `Tests/Agents/test_fleet_continuation.py`, which imports hypothesis at module level. The Console P0 gate's install line (shared by its three profiles) now adds `hypothesis`, and `test_console_p0_gate_runs_three_serial_minimal_python_312_profiles` pins the new line. No other package is missing: in run 44626839587 the sandboxed leg passed 230 and the restore leg 54, with no skips, and the admission leg passed 487; its only failures were the 9 hypothesis imports. Locally (hypothesis installed) the file passes; the admission leg itself is checked by the PR's run.

**AC#2 (leaked fake import).** Root cause: an autouse fixture (`_console_gateway_http_client_is_offline`) requests `monkeypatch` first, so the test's `monkeypatch.setattr("builtins.__import__", ...)` was undone only after every other autouse teardown (`--setup-plan`). `drain_test_app_user_data_dirs` tears down earlier and, when `Tests.UI.test_console_fleet_wake_wiring` is loaded, runs `from Tests.conftest import ...` under the one-argument fake. In shard 31 the wiring module was loaded by `test_console_launch_wake.py`'s import of `_attach_real_dbs`, which TASK-41507 put on the lane. Both tests now patch inside `monkeypatch.context()` around the call under test and assert the fake is gone afterwards. Red: the two files in one process (`-p no:xdist`, `-k` on the two tests) gave 6 passed, 6 errors with the CI message. Green: 6 passed; both whole files in one process: 15 passed.

**AC#3 (launch-wake teardown).** Root cause (product): `ConsoleFleetWakeCoordinator.dispose()` cancelled in-flight wake deliveries and cleared `_delivery_tasks`, so nothing awaited them. A cancelled `_deliver` swallows the cancellation and its `finally` still settles the attempt in AgentRuns (`_finish_attempt`: `read_attempt`, `snapshot`, `complete_wake`/`pause` through `run_owned_db_call`). `ConsoleRuntime._dispose_owned` then called `retire_agent_runs_db` while that SQL was running; `close_if_idle` saw a counted core operation or an acquisition on the file and refused (`sqlite_retirement_incomplete:AgentRuns`). The app's own quit only logs this ("SQLite retirement incomplete at quit"); the test teardown re-runs the dispose, which raises, and `retire_test_app_owners` leaves `_test_app_native_workers_uncertain` set, so `isolate_test_environment` refuses config retirement (the second sub-exception). A temporary diagnostic in `close_if_idle` and `retire_agent_runs_db` (reverted) recorded `core=True` and showed the delivery task at `_finish_attempt -> run_owned_db_call` in every refusal. Tests that build a controller rig had already joined the deliveries by hand (`_close_controller_rig`); the product did not.
Fix: `dispose()` keeps the cancelled deliveries in `_delivery_tasks` until each finishes (`_forget_delivery_task` already removes them), and `_dispose_owned` adds them to its bounded drain set before it retires AgentRuns. The retirement checks are unchanged.
New test `test_quit_settles_an_in_flight_launch_wake_before_retiring_agent_runs` (launch-wake file, already on the UI census): it parks the wake's turn in the stream, holds the settlement 0.5 s, and asserts no delivery is pending when `retire_agent_runs_db` runs and that the app's own shutdown task raised nothing. Red at base: "the quit retired AgentRuns while the cancelled wake delivery was still settling its attempt" (delivery at `console_fleet_wake.py:1365`). Whole file at -n 4, base product code: 3, 2 and 2 teardown ERRORs over three runs (load 13-19), plus the new test failing in the two runs that had it. With the fix: 19 passed in each of three runs (load 12-15). Serially, with the UI lane's flags: 19 passed in 164.6 s at load 15-17; the new test takes 9.4 s.

**Other checks.** Tests/CI -n 4: 1811 passed, 2 skipped. `test_module_size_ratchet.py`: 209 passed; `console_runtime.py` is raised 7078 -> 7083 by this change's own 5 lines. Covering fleet-wake, runtime, workflow and chat-start files at -n 4: 31 + 30 failures, all also red at base with the same failure messages (RecoveryRequired `raw_source_selection_changed`, workflow journeys). 3 Architecture failures (diagnostic inventory on `agent_service.py`, timer-path inventory on `project_index_binding_card.py`) are also red at base. Preflight passes. `ruff check tldw_chatbook --select F` (CI's scope) is clean; among the changed test files, `Tests/Chat/test_console_fleet_wake_safety.py:43` keeps an F401 that is already on dev. `ruff_gate.py` reports B023 37 > 36, which is already on dev: the touched modules have the same B/ASYNC/PLE findings at base and head and no B023.

Files: `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/UI/test_console_first_send_import_cycles.py`, `Tests/UI/test_console_launch_wake.py`, `Tests/Chat/test_console_fleet_wake_safety.py` (stale comment), `Tests/Architecture/test_module_size_ratchet.py`, `tldw_chatbook/Chat/console_fleet_wake.py`, `tldw_chatbook/Chat/console_runtime.py`, `backlog/docs/lessons-testing-evidence.md`.
