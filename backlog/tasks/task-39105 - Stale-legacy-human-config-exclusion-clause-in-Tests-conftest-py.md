---
id: TASK-39105
title: 'Stale legacy_human_config exclusion clause in Tests/conftest.py'
status: To Do
assignee: []
created_date: '2026-11-06 09:00'
labels:
  - tests
  - cleanup
dependencies: []
references:
  - Tests/conftest.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/conftest.py` around lines 1361-1368 keeps an exclusion clause for `legacy_human_config`. It is harmless today because the fixture no longer reselects config, but it exists on dev and misleads readers; this task's PR does not remove it. Found by the Roleplay B2b follow-up burn-down (2026-11-05), read on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The stale clause is removed and the tests that name legacy_human_config still pass
<!-- AC:END -->
