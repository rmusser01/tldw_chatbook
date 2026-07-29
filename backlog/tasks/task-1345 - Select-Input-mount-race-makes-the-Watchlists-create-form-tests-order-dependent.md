---
id: TASK-1345
title: Select/Input mount race makes the Watchlists create-form tests order-dependent
status: To Do
assignee: []
created_date: '2026-07-29 05:30'
labels:
  - watchlists
  - testing
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_watchlists_source_create_form.py` passes 3/3 in isolation but fails when run after
`Tests/UI/test_watchlists_content_pane.py`. Proven pre-existing rather than caused by Phase D:
with **all** of Phase D's new tests deselected, the create-form tests still fail 3/3 in that order.

Symptoms are a `Select`/`Input` mount race — `NoMatches` on `SelectCurrent`, and a truncated value
(`'orning' == 'Morning'`) indicating the input was read while still mounting.

The failures are intermittent across runs, so a green CI run is not evidence the race is gone.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The root cause of the mount race is identified and stated, not worked around with a sleep
- [ ] #2 The create-form tests pass regardless of the order the UI suite runs in, demonstrated by running them immediately after the content-pane suite
- [ ] #3 A deliberately re-introduced form of the race fails the tests, proving they discriminate it
<!-- AC:END -->
