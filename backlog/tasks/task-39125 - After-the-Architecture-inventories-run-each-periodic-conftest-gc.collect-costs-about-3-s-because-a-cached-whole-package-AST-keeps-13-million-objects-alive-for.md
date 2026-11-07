---
id: TASK-39125
title: "After the Architecture inventories run, each periodic conftest gc.collect() costs about 3 s, because a cached whole-package AST keeps 13 million objects alive for the rest of the session"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- ci
- performance
- tests
priority: low
references:
- Tests/conftest.py
- Tests/Architecture/test_timer_path_static_update_inventory.py
- Tests/Architecture/test_framework_armed_clock_inventory.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-39054
- TASK-39061
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and in the notes of TASK-39054 and TASK-39061 (round-11 branch fu/r11-arch). Line numbers are dev's.

**The cost.**
- `Tests/conftest.py:608` sets `_GC_EVERY = 25`, and the fixture runs `gc.collect()` every 25th test (`:666`).
- In the Architecture guards job, TASK-39054 measured trivial nodes' teardowns at about 3 s each.
- TASK-39061 measured `Tests/Utils/test_log_sanitizer.py` at 40.6 s, of which 31.8 s was this periodic collect (load 50-85).

**The cause.**
- `_load_package` (`Tests/Architecture/test_timer_path_static_update_inventory.py:2076`) is `@lru_cache(maxsize=1)` and keeps every module's AST plus its derived maps.
- `test_framework_armed_clock_inventory.py:205-212`, an early target of the Architecture guards job, calls it.
- Probes on dev (load 38-50) measured each collect:
  - with only the app imported: 42-49 ms (223k tracked objects);
  - after `_load_package()`: 2,962 modules, 13,195,206 tracked objects, 2.79-3.16 s;
  - after `cache_clear()`: 5 ms.
- Every later file in the same process pays about 3 s per 25 tests, which pushes the job toward its cap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 In a `--durations=0` run of the Architecture guards job's command, no trivial node's teardown exceeds 0.5 s (about 3 s before); before and after are recorded with load
- [ ] #2 `Tests/Utils/test_log_sanitizer.py`'s time inside that command is recorded against 40.6 s
- [ ] #3 Each inventory guard that uses the cached package still goes red on its own negative control
- [ ] #4 Three timings of the job's command are recorded
<!-- AC:END -->
