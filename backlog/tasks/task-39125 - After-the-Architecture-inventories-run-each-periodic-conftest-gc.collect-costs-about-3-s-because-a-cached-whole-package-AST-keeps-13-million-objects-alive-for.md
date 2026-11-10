---
id: TASK-39125
title: "After the Architecture inventories run, each periodic conftest gc.collect() costs about 3 s, because a cached whole-package AST keeps 13 million objects alive for the rest of the session"
status: Done
assignee:
- '@claude'
created_date: 2026-11-07 02:03
updated_date: 2026-11-10 18:11
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
- [x] #1 In a `--durations=0` run of the Architecture guards job's command, no trivial node's teardown exceeds 0.5 s (about 3 s before); before and after are recorded with load
- [x] #2 `Tests/Utils/test_log_sanitizer.py`'s time inside that command is recorded against 40.6 s
- [x] #3 Each inventory guard that uses the cached package still goes red on its own negative control
- [x] #4 Three timings of the job's command are recorded
<!-- AC:END -->

## Implementation Plan

1. Measure the Architecture guards job's exact command serially with `--durations=0`, before any change, with load.
2. Follow TASK-38702's pattern in the two modules that hold the cached package (`test_framework_armed_clock_inventory.py`, the job's caller, and `test_timer_path_static_update_inventory.py`, the cache's owner): a no-op `cleanup_file_descriptors` override (pure AST, nothing mounts an app, so no periodic collect walks the cached heap mid-module), and a module-scoped autouse fixture that clears `_load_package` and the caches derived from it when the module ends.
3. Negative controls: a temporary package edit (an indeterminate `ProgressBar`, an `auto_refresh` arming, and a timer-reachable `.update(` site) turns each guard that reads the cached package red; reverted.
4. Re-run the job's command three times with `--durations=0`; record the slowest teardown, `Tests/Utils/test_log_sanitizer.py`'s time, the totals and the load.

## Implementation Notes

**Approach.** `test_framework_armed_clock_inventory.py` (the job's only reader of `_load_package()`) now computes both of its censuses from one parse in `_package_censuses()` and calls `_load_package.cache_clear()` right after the load, so the ~13M-object package dies when the census returns, inside the census node's setup. `test_timer_path_static_update_inventory.py`, which owns the cache and needs it across its tests, follows TASK-38702's pattern: a no-op `cleanup_file_descriptors` override and a module-scoped autouse fixture that clears `_load_package`, `_package_graph` and `_package_clock_roots` when the module ends (it runs in no PR lane; this is for local `Tests/Architecture` runs).

**Deviation from the plan.** The first version gave the framework module the same release fixture. In its job run (load 54-92) the release itself, freeing 13M objects, showed as a 2.11 s teardown on the module's last, trivial node. Dropping the package inside the census moves that cost into the census node's setup; it also removes the module's second census pass (`test_no_unclassified_auto_refresh_arming`, 8.5-19.8 s call before).

**Measurements** (the job's exact command, extracted from `derived-artifacts.yml`, serial, plus `--durations=0 --durations-min=0`; local venv, 18 CPUs shared):

| Run | Code | Load (start -> end) | Wall | Result | Teardowns > 0.5 s | `test_log_sanitizer.py` (of which teardown) |
|---|---|---|---|---|---|---|
| base1 | dev ae9f9e8d21 | 72 -> 57 | 1259.7 s | 432 passed, 1 failed (runtime-policy node timed out, TASK-39126) | 16, max 10.67 s | 92.8 s (68.9 s) |
| base2 | dev ae9f9e8d21 | 25 -> 40 | 802.8 s | 433 passed | 16, max 11.16 s | 62.4 s (49.1 s) |
| final1 | this branch | 65 -> 37 | 694.3 s | 433 passed | 1: 0.53 s on `test_recovery_entrypoints.py::test_every_package_main_route_has_reviewed_classification` (a 45 s package scan, not trivial) | 15.4 s (2.0 s) |
| final2 | this branch | 37 -> 31 | 414.1 s | 433 passed | 0 (max 0.37 s) | 9.1 s (1.5 s) |
| final3 | this branch | 31 -> 28 | 370.5 s | 433 passed | 0 (max 0.18 s) | 8.2 s (0.9 s) |

The largest trivial-node teardown after the change was 0.44 s (final1, load about 65), against 3-11 s before. `test_log_sanitizer.py` was 40.6 s in TASK-39061's run (load 50-85); here it went from 62.4-92.8 s on dev to 8.2-15.4 s. The sum of all teardowns went from 104.7-136.8 s to 2.0-4.9 s. Part of the wall-time drop is TASK-39126's node (172.7-180 s -> 4.2-8.6 s). CI times are left to the coordinator.

**Negative controls** (temporary edits, all reverted; `git status` clean afterwards):
- One planted `tldw_chatbook/_r13a_planted.py` (an indeterminate `ProgressBar(total=None)`, `self.auto_refresh = 0.5`, and a `set_interval` callback calling `.update(`) plus a bogus row in each inventory, run as framework module then timer module in one process (so the timer census re-parsed after the framework census had dropped the shared cache): `test_every_framework_armed_clock_is_classified`, `test_no_unclassified_auto_refresh_arming` and `test_inventory_rows_are_not_stale` went red on the planted items; the timer guards `test_no_timer_path_update_defaults_to_layout_true`, `test_classified_sites_are_not_stale` and `test_clock_root_set_is_pinned` named the planted site, row and root.
- Renaming the splash screen's `set_interval` callback reference to `self._update_animation_r13a`: `test_census_actually_finds_the_known_clock_roots` and `test_clock_roots_all_resolve_loudly` went red.
- Pre-existing: those three timer guards are red on dev ae9f9e8d21 with the same command (3 failed, 14 passed, the dev file swapped in), on package drift (plugins screen, provider detection, approval card, project index card roots and sites). The module is in no PR lane; not changed here.

**Files.** `Tests/Architecture/test_framework_armed_clock_inventory.py`, `Tests/Architecture/test_timer_path_static_update_inventory.py`, `backlog/docs/lessons-testing-evidence.md` (a "Seen again" entry under TASK-38702's lesson).
