---
id: TASK-32386
title: 'Tests: test_closeout_single_app_route_cycle is red on dev'
status: To Do
assignee: []
created_date: '2026-09-11 10:30'
labels:
  - tests
  - library
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_library_adaptive_reader_closeout.py::test_closeout_single_app_route_cycle` fails on `origin/dev` itself, not on any wave branch -- it was A/B'd against a clean `git archive` of the base during the critique-10 wave (Tasks 6 and 7 both reported the identical failing name on base). The assertion looks for `library-browse-reader-shell` where the screen now mounts `.library-media-route`. A permanently red test on the default branch trains everyone to read a red file as noise, which is how a real regression gets waved through.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `Tests/UI/test_library_adaptive_reader_closeout.py` passes on origin/dev
- [ ] #2 The resolution states whether the screen or the assertion was wrong, rather than deleting the check
- [ ] #3 If the assertion was stale, it now names the selector the screen actually mounts
<!-- AC:END -->
