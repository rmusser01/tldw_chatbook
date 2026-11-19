---
id: TASK-42013
title: "A filtered-empty media focus test can miss the filter during a recompose"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:54
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
- [ ] #1 The test waits for the filter to be mounted, and passes 10 of 10 at `-n 4` with the load recorded
<!-- AC:END -->
