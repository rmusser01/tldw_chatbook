---
id: TASK-699
title: 'Library shell tests fail nondeterministically, hiding real regressions'
status: To Do
assignee: []
created_date: '2026-07-26 15:02'
labels:
  - testing
  - library
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Library shell test file fails around six tests on every run, but not the same six: the set shifts between runs on identical code, and every one of the varying tests passes when run alone. This holds equally on the development branch, so it is not caused by any single change. The cost is that the file cannot answer whether a change broke something -- a genuine regression would be indistinguishable from the day's shuffle, and comparing failure counts between runs actively misleads. Three failures are stable and separately actionable; the rest are order-dependent, clustering around note conflict handling, note save results after a switch, export registry warnings, and ingest canvas isolation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Library shell test file produces the same result on repeated runs of the same code,A test that passes alone does not fail as part of the file,The three consistently failing tests are either fixed or explicitly recorded as known failures with reasons
<!-- AC:END -->
