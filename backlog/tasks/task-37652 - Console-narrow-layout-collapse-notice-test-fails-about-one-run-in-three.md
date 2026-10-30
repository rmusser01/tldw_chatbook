---
id: TASK-37652
title: Console narrow-layout collapse-notice test fails about one run in three
status: To Do
assignee: []
created_date: '2026-10-29 13:53'
labels:
  - console
  - tests
  - flake
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seen while gating TASK-33626.1/.2 on 2026-10-29. Tests/UI/test_console_narrow_layout.py::test_responsive_collapse_posts_notice_once_per_rail failed 2 of 6 isolated runs on origin/dev 7b9fb3d555 and 2 of 5 on the TASK-33626 branch, always with 'expected a Context collapse notice, got []'. Failing runs finish about 0.7 s sooner than passing ones, which suggests the assertion reads the notices before the width rule's collapse has posted one. A gate test that fails a third of the time teaches people to rerun reds instead of reading them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes 20 consecutive isolated runs on a loaded host
- [ ] #2 The test waits for the collapse it asserts on (or the notice) instead of a fixed number of pauses
- [ ] #3 If the flake turns out to be a product race (a notice that is sometimes never posted), the product is fixed and the test fails on the pre-fix code
<!-- AC:END -->
