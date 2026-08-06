---
id: TASK-2560
title: Schedules destination visual-parity tests fail on a missing media DB
status: To Do
assignee: []
created_date: '2026-08-06'
labels:
  - tests
  - scheduling
dependencies: []
priority: medium
---

## Description (the why)

`Tests/UI/test_destination_visual_parity_correction.py` has four failing
cases on the **schedules** destination, each raising `Local media DB is
required...`. Confirmed pre-existing on dev and unrelated to the Watchlists
UAT work: found during batch-4 verification only because that file entered
the verification set for the first time there, and reproduced in isolation.

A test that fails on every run teaches the suite's readers to ignore
failures, which is how the `_delete_item` case (task-2330) survived as long
as it did.

## Acceptance Criteria (the what)

- [ ] The four schedules cases either provide the media DB the destination
      genuinely needs, or are skipped with an explicit reason naming the
      dependency.
- [ ] The whole file passes on dev.
- [ ] If the failure reflects a real product requirement (a destination that
      cannot render without a media DB), that requirement is stated in the
      test rather than implied by a crash.
