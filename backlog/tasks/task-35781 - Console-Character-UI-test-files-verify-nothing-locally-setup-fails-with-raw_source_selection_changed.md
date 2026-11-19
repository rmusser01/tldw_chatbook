---
id: TASK-35781
title: >-
  Console Character UI test files verify nothing locally: setup fails with
  raw_source_selection_changed
status: To Do
assignee: []
created_date: '2026-10-20 06:01'
labels:
  - testing
  - console
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four Tests/UI files that drive the real Console Character display error at fixture setup on a developer machine with RecoveryRequired: raw_source_selection_changed: test_console_character_context.py (16 errors), test_character_context_async_ownership.py (16), test_character_navigation_screen_reuse.py (7) and test_console_progress_navigation_owners.py (4). The errors appear identically on dev and on feature branches, so for Character display changes (for example TASK-33620.15.6) these files are compared by test id and confirm nothing locally; the main UI-level Character test file is effectively unexercised outside CI. lessons-testing-evidence.md records the cause class: a plain gated Tests/UI test that reaches a config read needs a private profile. These files should run under a private profile so they verify behaviour locally as well as in CI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The four named files run under the plain pytest command on a developer machine with no RecoveryRequired setup errors
- [ ] #2 Each file's test count and pass/fail outcome in CI is unchanged or better
- [ ] #3 No file is moved into a slower lane and no test is skipped to achieve this
<!-- AC:END -->
