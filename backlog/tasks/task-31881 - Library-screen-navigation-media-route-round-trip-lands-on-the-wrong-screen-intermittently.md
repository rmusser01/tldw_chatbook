---
id: TASK-31881
title: >-
  Library screen-navigation media route round-trip lands on the wrong screen
  intermittently
status: To Do
assignee: []
created_date: '2026-09-06 18:13'
labels:
  - library
  - tests
  - flaky
  - pre-existing-red
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_screen_navigation.py::test_media_route_round_trips_to_the_library_media_row fails intermittently with assert 'Screen' == 'LibraryScreen' -- the route lands on a bare Screen instead of LibraryScreen. It reproduces on trees that do not contain the Library media extraction, so it is a dev-side route-landing race, not extraction fallout. It belongs to the same test_screen_navigation.py family the Library-decomposition recipe has been documenting as run-to-run flakiness since wave 3; this task is to find the actual race rather than keep re-dispositioning the name every wave.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The mechanism of the wrong-screen landing is identified and stated (what wins the race against the route push, and why the assertion can observe the intermediate screen)
- [ ] #2 test_screen_navigation.py::test_media_route_round_trips_to_the_library_media_row passes 10 of 10 isolated single-node runs
- [ ] #3 The sibling route-landing names the recipe records for this file are re-measured after the fix and their status recorded (fixed by the same cause, or still flaky for a different one)
- [ ] #4 recipe backlog/docs/library-decomposition-recipe.md section 7's entry for this test is updated or removed to match the outcome
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Filed by the wave-7 (media) Library-decomposition wave close. Measured rates, single-node isolated runs:

  branch (wave-7 tip)            5 failed / 5 runs
  isolated parent 78186d159      2 failed / 3 runs

Same assertion on both trees, so pre-existing rather than wave-caused. In the batch that surfaced it, a RAG-named sibling (test_search_route_lands_on_library_rag_canvas) was the BASELINE-unique failure while this media-named one was the branch-unique -- the bidirectional signature recipe section 7 calls independent evidence of run-to-run flakiness. Related documented names in the same file: test_search_route_round_trips_to_the_library_rag_row, test_library_screen_round_trip_returns_to_landing_with_rag_draft, test_generic_library_entry_lands_hub_on_first_visit. Provenance: wave-7 task 3 report section 9.1 (.superpowers/sdd/2026-09-06-library-decomposition-wave7-media/task-3-report.md).
<!-- SECTION:NOTES:END -->
