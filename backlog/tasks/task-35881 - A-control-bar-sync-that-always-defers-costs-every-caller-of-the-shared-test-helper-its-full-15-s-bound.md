---
id: TASK-35881
title: >-
  A control-bar sync that always defers costs every caller of the shared test
  helper its full 15 s bound
status: To Do
assignee: []
created_date: '2026-10-21 00:35'
labels:
  - console
  - tests
  - ci
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Since TASK-35660, the shared test helper _sync_console_control_bar_now (Tests/UI/test_console_native_chat_flow.py) retries a deferred control-bar sync for up to 15 s before it fails, naming the last refusal. That bound is right for one caller on a loaded host. But a product regression that makes every sync defer would cost each of the helper's roughly 70 callers the whole 15 s before it fails: about 17 minutes spread across the UI Fast Lane shards, whose jobs have 20 and 35 minute limits, so the shards could be cancelled at their limit instead of reporting the regression. Raised by TASK-35660's independent review on 2026-10-20.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A regression that makes every control-bar sync defer (scratch mutation, recorded in the notes) adds at most about one bound per pytest process, not one per caller, and every affected caller still fails
- [ ] #2 The first caller's failure message still names the last refusal and Send's state; later callers name that earlier failure rather than passing or failing without a reason
- [ ] #3 A single slow caller on a loaded host still gets the whole bound before it fails
<!-- AC:END -->
