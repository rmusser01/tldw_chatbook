---
id: TASK-15668
title: 'Repair the two Settings workspaces-category test failures (unowned dev red)'
status: To Do
assignee: []
created_date: '2026-08-11 21:30'
labels:
  - settings
  - tests
  - baseline
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two tests in `Tests/UI/test_settings_workspaces_category.py` fail and were verified pre-existing at commit 2e26bbcad, i.e. not introduced by the supervisor-fleet work. They are unowned. Leaving them red erodes the value of the whole UI suite as a gate, which is the same argument task-3070 was filed on.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both tests pass on dev, or are removed with a recorded reason if the behaviour they assert is gone
- [ ] #2 The diagnosis names which commit changed the behaviour, established from history rather than assumed
- [ ] #3 Tests/UI/test_settings_workspaces_category.py runs with zero failures
<!-- AC:END -->
