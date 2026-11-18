---
id: TASK-41712
title: "test_on_screen_suspend_stops_every_timer_in_isolation is red on dev: its hand-built screen lacks _artifacts_navigation"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
labels:
- follow-up
- tests
- library
priority: low
references:
- Tests/UI
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41500 and confirmed by its review (round 16): the test's hand-built Library screen has no `_artifacts_navigation`, so it fails with `AttributeError` on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test builds the screen state the suspend path reads and passes
- [ ] #2 It runs on a PR lane, or the notes say why not
<!-- AC:END -->
