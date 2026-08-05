---
id: TASK-2330
title: check-now-failure test asserts a method the screen no longer defines
status: To Do
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - tests
dependencies: []
priority: medium
---

## Description (the why)

`Tests/UI/test_watchlists_check_now_failure.py` has a parametrized case
(`[_delete_item]`) that fails on dev: it asserts against a method the screen
no longer defines (`_delete_item` was removed when the `d` binding was routed
through the TASK-1541 drain in PR #1342). Found during UAT batch-2 review and
verified failing on `origin/dev` at the batch-2 merge base (`ab9105c9d`) —
pre-existing there, not introduced by any open branch.

## Acceptance Criteria (the what)

- [ ] The parametrized case either targets the real current writer path or is
      removed with the reasoning recorded.
- [ ] The whole file passes on dev.
- [ ] The exemption contract the file documents (debug-level loader logging
      requires a failure toast) still has a discriminating case per loader.
