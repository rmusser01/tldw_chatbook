---
id: TASK-39054
title: "The undefined-private-helper guard runs one node per module, so in the Architecture job its 2,943 nodes cost about six minutes of periodic GC teardowns for about 30 s of AST work"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 10:09
updated_date: 2026-11-06 17:59
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

Files: `Tests/Architecture/test_no_undefined_module_globals.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.

**Architecture guards job timings (shared by TASK-39054, TASK-39055, TASK-39060 and TASK-39061).**
- Before adding anything: CI ran the step in 402 s (job 7 m 38 s) on #3436's run 42218508176, 2026-11-06.
- The step's command, run with `bash $FU/jobstep.sh fu-r11-arch "Run Architecture guard contracts"` (serial, `--timeout=180 --tb=short`). Each run of the final 36 targets gave 383 passed:

| Command | Targets | Nodes | pytest time | Load |
|---|---|---|---|---|
| Base 2684e5453f | 29 | 160 | 153.9 s | 7-11 |
| Base 2684e5453f | 29 | 160 | 457.0 s | 27-45 |
| Base 2684e5453f | 29 | 160 | 283.9 s | 32-38 |
| This branch, run 1 | 36 | 383 | 563.4 s | 25-48 |
| This branch, run 2 | 36 | 383 | 451.2 s | 24-48 |
| This branch, run 3 | 36 | 383 | 440.6 s | 24-57 |
| This branch, right after the 283.9 s base run | 36 | 383 | 413.8 s | 32-48 |

- Six agents shared the machine, and the load swung from 24 to 57, so the totals mostly measure the load.
- One more run, with `--durations=0` (862.4 s at load 50-85), attributes setup, call and teardown time per file. The added files cost 146.9 s of its 858.3 s:
  - `test_no_undefined_module_globals.py`: 72.5 s for 3 nodes, with no periodic-GC teardown;
  - `test_log_sanitizer.py`: 40.6 s for 168 nodes, of which 31.8 s is the conftest's every-25th-test `gc.collect()`;
  - the four ChatScreen files: 33.5 s for 51 nodes, 8.9 s of it GC;
  - the two boundary nodes: 0.3 s.
- An extra job run overlapped my temporary source reverts for a boot-census bisect (12 failed). I discarded it.
