---
id: TASK-32105
title: >-
  Tests: `Tests/UI/test_library_prompts_canvas.py` is flaky in both directions
  under host load
status: To Do
assignee: []
created_date: '2026-09-08 22:42'
labels:
  - tests
  - library
  - prompts
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During the critique-8 wave two full runs of this file on the same pair of commits disagreed in both directions (five names failed on base and passed on branch, two the other way); name-set diffs against it require a quiet machine, and it took 85 minutes under load. Found while verifying PR #2528. Rider from the critique-8 fix wave reviews (plan Docs/superpowers/plans/2026-09-08-library-crit8-wave.md; wave PRs #2519 #2523 #2524 #2525 #2528 #2531).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The flaky names are identified and either stabilised or marked with the reason
- [ ] #2 A full run of the file on a loaded host gives the same failing-name set twice
<!-- AC:END -->

## Evidence

<!-- Appended by task-32461 fix round 1 (2026-09-14) -- three full runs of this
file on ONE tree, by two different sessions, disagreeing in both directions. -->

Full-file runs (`-q -p no:cacheprovider --timeout=300`), same checkout:

    implementer, branch    20 failed / 322 passed   701.55s
    implementer, baseline  18 failed / 324 passed   711.07s   (3-line diff, none of it reachable by these tests)
    reviewer,   branch     17 failed / 325 passed   664.27s   (identical tree to run 1)

16 of the reviewer's 17 names are in the implementer's baseline set. Names that
moved between runs on the identical tree:

    test_library_prompt_compatibility_editor_discard_returns_to_current_list
    test_library_prompt_open_existing_button_shows_only_in_name_in_use_state_and_opens_it
    test_library_prompt_conflict_save_as_new_replaces_source_history_identity
    test_library_prompt_pager_first_and_filter_failure_states[size0]

All three of the first group pass in isolation on the branch
(`3 passed` in one run), which is AC#2's disagreement in its cleanest form:
same file, same tree, three runs, three different failing-name sets.
