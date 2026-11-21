---
id: TASK-42144
title: "The test CSS parse cache grows without bound and makes every later gc.collect() slow"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 19:59
labels:
- follow-up
- tests
- performance
priority: low
references:
- Tests/UI/css_cache.py
- Tests/conftest.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the skillsrec stream (TASK-42073, round 18). `Tests/UI/css_cache.py` keeps a session-wide `_GLOBAL_PARSE_CACHE` (line 27) that is never bounded or reset. Its key includes the stylesheet's variables (lines 63-70), so each theme adds a full parse of the app stylesheet.

TASK-42073 measured it: after the theme-contrast node's 89 themes, the process held about 10.6M gc-tracked objects (373k before it), mostly parsed CSS (516k `RuleSet`, 1.1M `Selector`, 515k `Styles`). `Tests/conftest.py::cleanup_file_descriptors` runs a full `gc.collect()` after every test under an app-mounting directory, and each collect then took about 5.5 s. With `TLDW_TEST_CSS_CACHE=0`, that node's teardown fell from 5.8 s to 0.5 s, and the next Skills tests' teardowns from 5.3 s to 0.11 s.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cache is bounded, or reset where a theme sweep ends, so that after the theme-contrast node each of the next tests' teardowns in the same process takes under 1 s (TASK-42073 measured 0.11 s with the cache off, and 5.3-5.4 s with it on)
- [x] #2 The notes compare those teardown times with the cache on, before and after the change, in one process
- [x] #3 Repeated parses of the same stylesheet with the same variables still hit the cache, and a test pins it
<!-- AC:END -->

## Implementation Plan

1. Measure in one process, cache on, at base: the theme-contrast node then a few app-mounting tests, `--durations=0`, teardown times after the node.
2. Bound `Tests/UI/css_cache.py`'s global cache the way the app's own `TieAwareStylesheet` does: an LRU of per-variables caches, a few variable sets kept.
3. Re-measure the same command; teardowns after the node under 1 s.
4. Add a test that a repeated parse with the same variables hits the cache (and a test that the LRU evicts), in a laned file.

## Implementation Notes

- **Fix.** `Tests/UI/css_cache.py` now keys an `OrderedDict` LRU by the variables fingerprint, each holding a dict keyed by Textual's own parse key, and keeps the latest `_VARIABLE_SETS_KEPT = 2` variable sets, the shape of the app's `TieAwareStylesheet` (which keeps 4). A stdlib `OrderedDict`, not `textual.cache.LRUCache`, so `cache_info()` can still count entries. The rest is unchanged: same parse key, same list copies, same `TLDW_TEST_CSS_CACHE=0` hatch.
- **Why 2.** I recorded every cached parse (variables, key, hit/miss, parse time) over a serial run of 25 census entries with the theme node in the middle, then replayed it against an LRU of K sets. Outside the sweep, 11,771 of 14,059 accesses used one variable set. Unbounded: 2,382 misses, 108 s of parsing. Every K from 1 to 16: 2,459 misses (+77, about 2 s), because the sweep's 88 other sets push the main one out once, whatever K is. Each kept set holds about 115k objects. With the theme node then a probe test in one process, objects after `gc.collect()` and the time of the next collect:
  - base: 10,566,297 objects, 6.30 s (load 52);
  - K=4: 757,828, 0.78 s (load 31);
  - K=2: 526,582, 0.26 s (load 33);
  - cache off: 295,330, 0.13 s (load 30).
- **Teardowns after the node (AC#1, AC#2).** One process, cache on: the theme-contrast node, then 8 quick `Tests/Skills/test_skills_import.py` tests (Tests/Skills collects after every test). The same command each time, `-p no:xdist --durations=0`:
  - before (unbounded): 7.28-14.28 s each (node call 89 s, load 35-51);
  - after, K=2, three runs: 0.40-1.18 s (load 43-57), 0.29-0.73 s (load 33-51) and 0.27-0.43 s (load 37-50). The one value over 1 s was in the first run at load 56. Collection alone costs 0.26 s at K=2, so the rest is the machine's load;
  - for reference: K=4 0.59-0.90 s (load 44-56), cache off 0.11-0.18 s (load 40-46).
  - In the 25-entry trace run (base cache, load 45-53), the tests before the node tore down in 0.4-1.9 s and those after it in 12-28 s.
- **Test (AC#3).** New `Tests/CI/test_css_parse_cache.py`, which the fast lane runs through its `Tests/CI` target (no census edit). `test_a_repeat_with_the_same_variables_hits_the_cache` parses a unique CSS twice on two fresh `Stylesheet`s with the same variables and expects one real `parse()`, and a second one for other variables. `test_only_the_latest_variable_sets_are_kept` parses K+1 sets and expects the newest to hit and the oldest to parse again.
  - Red: the base cache plus the constant alone failed the second test with `the oldest variable set is dropped`. A temporary always-miss lookup failed both: `assert 2 == 1` and `the newest variable set is kept`. Both reverted.
  - Green: 2 passed.
- **Lane comments (coordinator-authorized follow-up).** The comments on admission-sensitive suites 3 in `.github/workflows/derived-artifacts.yml` and on `ADMISSION_3_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py` gave the unbounded cache as the reason the Skills file runs first. They now say the cache is bounded, so the order no longer matters for teardown cost. Only comments changed; the target order stays as measured. Tests/CI at -n 4: 1815 passed, 2 skipped.
- Files: `Tests/UI/css_cache.py`, `Tests/CI/test_css_parse_cache.py`, and comments in `.github/workflows/derived-artifacts.yml` and `Tests/CI/test_ci_queue_pressure_contract.py`.
