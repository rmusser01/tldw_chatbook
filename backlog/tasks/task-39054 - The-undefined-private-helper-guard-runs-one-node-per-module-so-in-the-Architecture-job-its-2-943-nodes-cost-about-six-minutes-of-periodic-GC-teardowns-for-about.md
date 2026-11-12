---
id: TASK-39054
title: "The undefined-private-helper guard runs one node per module, so in the Architecture job its 2,943 nodes cost about six minutes of periodic GC teardowns for about 30 s of AST work"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 10:09
updated_date: 2026-11-07 01:14
labels:
- follow-up
- tests
- architecture
- ci
priority: medium
references:
- Tests/Architecture/test_no_undefined_module_globals.py
- Tests/conftest.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-38989
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on Architecture PR #3436 (head b7c8e809b3), which lands TASK-38989; the #3436 timings below were taken on an earlier commit of that branch. Line numbers are dev's; ones marked #3436 are from that head. TASK-38989's notes propose this and keep the file off its new Architecture guards job. No PR lane runs it on dev or on #3436.

**Shape and cost.**
- `Tests/Architecture/test_no_undefined_module_globals.py:158` (#3436 :176) parametrizes once per module: `@pytest.mark.parametrize("path", _module_files(), ids=lambda p: str(p.name))`. That gives 2,942 nodes on dev and 2,943 on #3436. The ids are basenames, which repeat.
- Run alone (serial, #3436): 2,943 passed in 37.0 s, load 18-20. The AST check itself is about 30-33 s, 10.5 s of it parsing.
- In the Architecture guards job's command (27 targets, serial, `--timeout=180 --tb=short`, #3436):
  - without this file: 152 passed in 210.5 s (load 12-18);
  - with it: 3,095 passed in 567.1 s (load 12-20).
  - So the file adds about 357 s.

**Why the context matters.**
- `Tests/conftest.py:608` sets `_GC_EVERY = 25`, and `cleanup_file_descriptors` runs `gc.collect()` every 25th test (`:632`, the collect at `:666`).
- Once earlier files have loaded the app heap, each collect takes about 3 s. The 15 slowest durations in the combined run were all 3.0-3.4 s teardowns of trivial nodes.
- About 118 collects of about 3 s each is about 354 s, which matches the added time. One node removes almost all of it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The guard is one node that checks every module and names every offending module and helper in one failure message; the file's other tests stay
- [x] #2 Negative control: a temporarily deleted private helper turns the guard red, and the message names its module
- [x] #3 The file runs on the Architecture guards PR job (never the fast PR contract); that job's command is timed three times with it, and the added time is recorded against 210.5 s without the file and 567.1 s with today's per-module file
<!-- AC:END -->

## Implementation Plan

1. Replace the per-module parametrization with one node that walks every module and collects each offender as `<module>: [helpers]`, failing once with the whole list. Keep the exemption list, the stale-exemption test and the class-body test.
2. Negative control: temporarily delete a private helper's definition in two modules; the one node fails and names both modules and helpers. Revert.
3. Append the file to the Architecture guards step and to `ARCHITECTURE_GUARD_TARGETS`, then time the job command three times and record the added time against 210.5 s (without the file) and 567.1 s (with the per-module file).

## Implementation Notes

**Status: Done.**

- AC#1: `test_every_private_helper_called_in_a_module_is_defined_in_it` (`Tests/Architecture/test_no_undefined_module_globals.py`) is now one node. It walks `_module_files()` once, skips `_EXEMPT` rows and unparseable files as before, and collects each offender as `<module>: [helpers]`. It then fails once, listing all of them. The stale-exemption test and the class-body test are unchanged, and the unused `pytest` import is gone. Collected nodes: 2,943 at base, 3 now.
- AC#2, negative control (temporary, reverted): I renamed `_line_end` in `Utils/log_sanitizer.py` and `_ratio` in `Evals/research_report_scorer.py`. The node failed with a message that named both: `tldw_chatbook/Evals/research_report_scorer.py: ['_ratio']` and `tldw_chatbook/Utils/log_sanitizer.py: ['_line_end']`. (A first attempt renamed `path_validation._logging_logger`, which runs at import, so collection failed before the guard ran.) Green after the revert: 3 passed in 17.7 s, serial, load about 10.
- AC#3: the file is appended to the Architecture guards step and to `ARCHITECTURE_GUARD_TARGETS`. Added time compared with the task's figures:
  - The per-module file used to add 357 s to the job command: 210.5 s without it, 567.1 s with it, load 12-20, on #3436.
  - The one node now costs 17.7 s alone at load about 10, and 72.5 s in a full job run at load 50-85. That run shows no periodic-GC teardown for the file: 3 nodes never reach the 25-test cadence.
  - So the saving is about 285-340 s of job time, and the guard now runs on a PR lane for the first time.
  - The job command's three timed runs are in the table below.

- Review r11a-rev1 fix (Minor): the one-node guard used to drop an unparseable module silently with `except SyntaxError: continue`, where the per-module form had shown a visible skip.
  - The scan now lives in `_guard_report(package)`, which returns `(offenders, unparseable)`. The guard fails and names every module that does not parse, with its line and message, alongside the offenders.
  - New negative control: `test_guard_names_unparseable_and_offending_modules` builds a scratch package containing `broken.py`, `caller.py` (which calls a missing `_gone`) and `clean.py`, and asserts that both bad modules are named.
  - With the old silent skip temporarily restored, the control failed. With the fix the file passes 4 of 4, serial.

Files: `Tests/Architecture/test_no_undefined_module_globals.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.

**Architecture guards job timings (shared by TASK-39054, TASK-39055, TASK-39060 and TASK-39061; corrected per review r11a-rev1).**

- **CI before this branch:** the step took 402 s (job 7 m 38 s) on #3436's run 42218508176, 2026-11-06.
- **The step at each commit:**

| Commit | Targets | Nodes | Why |
|---|---|---|---|
| Base | 29 | 160 | |
| 77c5a7a085 | 36 | 383 | Two boundary nodes listed separately. |
| d29bb13b6a | 35 | 387 | The boundary file runs whole (6 nodes). |
| This fix | 34 | 383 | The wave-6 closeout file folded into the realtime test: 1 target and 5 nodes fewer. The guard's new unparseable-module control adds 1 node. |

- **Current figure (review r11a-rev1's run at d29bb13b6a):** 386 passed plus 1 pre-existing timeout, in 1,073.5 s at load 84-91. The timeout is `test_profile_owned_path_inventory::test_compatibility_and_runtime_policy_constants_have_no_new_runtime_owners`, which also hits 180 s at base under load. Of that time, this branch's additions cost 149.2 s.
- **Earlier runs:** my own runs of the step's command, through `jobstep.sh`, serial, pytest-reported time. The load swung from 24 to 57, so the totals mostly measure the load.

| Command | Targets | Nodes | Result | pytest time | Load |
|---|---|---|---|---|---|
| Base 2684e5453f | 29 | 160 | 160 passed | 153.9 s | 7-11 |
| Base 2684e5453f | 29 | 160 | 160 passed | 457.0 s | 27-45 |
| Base 2684e5453f | 29 | 160 | 160 passed | 283.9 s | 32-38 |
| 77c5a7a085, run 1 | 36 | 383 | 383 passed | 563.4 s | 25-48 |
| 77c5a7a085, run 2 | 36 | 383 | 383 passed | 451.2 s | 24-48 |
| 77c5a7a085, run 3 | 36 | 383 | 383 passed | 440.6 s | 24-57 |
| 77c5a7a085, right after the 283.9 s base run | 36 | 383 | 383 passed | 413.8 s | 32-48 |

- **Where the time goes:** a `--durations=0` run at 77c5a7a085 (load 50-85) attributes the additions as follows:
  - `test_no_undefined_module_globals.py`: 72.5 s, with no periodic-GC teardown;
  - `test_log_sanitizer.py`: 40.6 s, of which 31.8 s is the conftest's every-25th-test `gc.collect()`;
  - the ChatScreen files: 33.5 s;
  - the boundary nodes: 0.3 s.
