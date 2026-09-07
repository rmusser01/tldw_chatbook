---
id: TASK-31947
title: >-
  Perf Guard - the UI-ready module census pin has been red on dev since the
  Inspect rail redesign
status: To Do
assignee: []
created_date: '2026-09-07 08:25'
labels:
  - library
  - media-ux
  - test-debt
  - perf
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seen while landing media wave-5 PR F (2026-09-05): the 'UI latency guardrails' workflow's Tests/Performance/test_ui_ready_module_census.py::test_ui_ready_module_census_stays_at_the_pinned_size has been red on dev since 7e904737c (Inspect rail Environment redesign) and 5f12507c1 (#2414 library reuse). It is not a Library media failure - the boot import set grew and nobody re-pinned or trimmed it, so every PR since reads a red Perf Guard it did not cause.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The census pin is green on dev, either by trimming the boot imports back or by re-pinning with the reason recorded in the pin
- [ ] #2 The change(s) that grew the boot import set are named in this task or in the pin's comment
<!-- AC:END -->
