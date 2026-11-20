---
id: TASK-42013
title: "A filtered-empty media focus test can miss the filter during a recompose"
status: Done
assignee: ['@claude']
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 16:37
labels:
- follow-up
- tests
- library
priority: low
references:
- Tests/UI/test_library_shell.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41711 (round 17). `test_background_recompose_restores_focus_on_a_filtered_empty_media_list` waits for an empty page (no rows, no recovery buttons), which also holds during a recompose's remove-then-mount gap, so the next `query_one("#library-media-filter")` can raise `NoMatches` under load.

Related: TASK-41766 covers the same wait-then-`query_one`-mid-recompose shape for five Media tests (its AC#4 fixes `_wait_for_selector`); TASK-32292 covers this test's docstring.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test waits for the filter to be mounted, and passes 10 of 10 at `-n 4` with the load recorded
<!-- AC:END -->

## Implementation Plan

1. Read the test and the TASK-41766 helper scope so as not to take it over.
2. After the empty-page wait, wait (bounded) for `#library-media-filter` to be mounted before querying it.
3. Run the test 10 times at `-n 4` and record the load.

## Implementation Notes

The empty-page wait in `test_background_recompose_restores_focus_on_a_filtered_empty_media_list` (`Tests/UI/test_library_shell.py`) now also requires `#library-media-filter` to be mounted and not `_pruning`. `_wait_for_condition` treats the `NoMatches` of a missing filter as "not yet", and it returns straight after a true predicate, so the `query_one` that follows reads the same tree. The assertions are unchanged. `_wait_for_selector` (TASK-41766's AC#4) and the five TASK-41766 tests are untouched.

- **Gap evidence** (temporary probe, deleted): after the filter-miss page settled, the probe recomposed the Media canvas three times and read both waits on every loop turn (`asyncio.sleep(0)`, as when `pilot.pause`'s screen wait gives up under load). The old condition held with no live filter on 38 of 180 turns, in 2 of 2 runs (states seen: no filter, a pruning filter, a filter not yet mounted). The new one held on 0. Polled with `pilot.pause(0)`, neither ever held in the gap, because the pause waits for the screen to drain; that is why the miss needs load.
- **10 of 10:** the test, temporarily parametrized 10 ways and run at `-n 4`, passed 10 of 10 (load 26.8 at start, 33.1 at end). The parametrization was reverted.
- The file runs whole, split by node, in the `derived-artifacts.yml` Library shell lane, so no census edit is needed.

Files: `Tests/UI/test_library_shell.py`.
