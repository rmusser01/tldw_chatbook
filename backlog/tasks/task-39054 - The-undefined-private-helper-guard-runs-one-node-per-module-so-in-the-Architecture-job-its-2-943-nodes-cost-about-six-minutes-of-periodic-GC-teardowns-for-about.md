---
id: TASK-39054
title: "The undefined-private-helper guard runs one node per module, so in the Architecture job its 2,943 nodes cost about six minutes of periodic GC teardowns for about 30 s of AST work"
status: To Do
created_date: 2026-11-06 10:09
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
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on the Architecture branch that lands TASK-38989 (the state after it merges). TASK-38989's notes propose this and keep the file off its new Architecture guards job. No PR lane runs it on dev or on the branch.

**Shape and cost.**
- `Tests/Architecture/test_no_undefined_module_globals.py:158` parametrizes once per module: `@pytest.mark.parametrize("path", _module_files(), ids=lambda p: str(p.name))`. That gives 2,942 nodes on dev and 2,943 after the merge. The ids are basenames, which repeat.
- Run alone (serial, after the merge): 2,943 passed in 37.0 s, load 18-20. The AST check itself is about 30-33 s, 10.5 s of it parsing.
- In the Architecture guards job's command (27 targets, serial, `--timeout=180 --tb=short`, after the merge):
  - without this file: 152 passed in 210.5 s (load 12-18);
  - with it: 3,095 passed in 567.1 s (load 12-20).
  - So the file adds about 357 s.

**Why the context matters.**
- `Tests/conftest.py:608` sets `_GC_EVERY = 25`, and `cleanup_file_descriptors` runs `gc.collect()` every 25th test (`:656-666`).
- Once earlier files have loaded the app heap, each collect takes about 3 s. The 15 slowest durations in the combined run were all 3.0-3.4 s teardowns of trivial nodes.
- About 118 collects of about 3 s each is about 354 s, which matches the added time. One node removes almost all of it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The guard is one node that checks every module and names every offending module and helper in one failure message; the file's other tests stay
- [ ] #2 Negative control: a temporarily deleted private helper turns the guard red, and the message names its module
- [ ] #3 The file runs on the Architecture guards PR job (never the fast PR contract); that job's command is timed three times with it, and the added time is recorded against 210.5 s without the file and 567.1 s with today's per-module file
<!-- AC:END -->
