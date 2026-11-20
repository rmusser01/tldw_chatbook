---
id: TASK-42028
title: "Three required checks are red on dev a6df4d8d48 from round 16"
status: To Do
assignee: []
created_date: 2026-11-20 01:20
updated_date: 2026-11-20 01:20
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
- [ ] #1 The admission-sensitive Console P0 job runs every test in test_console_agent_call_thinking.py with no import error
- [ ] #2 test_console_first_send_import_cycles.py passes with no teardown error when it shares an xdist worker with test_console_fleet_wake_wiring.py, and its fake import cannot leak past the test body
- [ ] #3 test_console_launch_wake.py has no teardown error in repeated -n 4 runs; the AgentRuns database a launch wake opens is retired before teardown, with the root cause named in the notes
<!-- AC:END -->
