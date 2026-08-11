---
id: TASK-15111
title: >-
  Tests/UI reaches a live localhost:8080 endpoint when one is listening
status: To Do
assignee: []
created_date: '2026-08-11 04:00'
labels:
  - tests
  - test-infrastructure
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed while repairing task-14920: the `Tests/UI` console suites **make real requests to `127.0.0.1:8080/v1/models`** when something happens to be listening on that port on the developer's machine.

That makes those suites environment-dependent in the worst way — they behave differently depending on whether an unrelated local server is running, and the difference is silent. A developer with a local model server up is running different tests from CI, and neither knows it. It is also the mirror image of two traps this repo has already recorded: a missing optional extra faking a code regression, and a green suite saying nothing about installs that are not yours.

Worth checking as part of this: whether any test can *mutate* state on a live endpoint it reaches, and whether the same pattern exists outside `Tests/UI`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 No test in the suite performs network I/O to a live local endpoint; the boundary is stubbed or blocked
- [ ] #2 A guard makes the escape impossible to reintroduce silently (e.g. sockets blocked by default in the UI test configuration, with an explicit opt-in for any test that genuinely needs one)
- [ ] #3 The check covers whether any such call could mutate state on the endpoint it reached, not just read from it
<!-- AC:END -->
