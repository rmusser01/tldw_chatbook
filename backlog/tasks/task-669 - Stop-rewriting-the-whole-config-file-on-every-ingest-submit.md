---
id: TASK-669
title: Stop rewriting the whole config file on every ingest submit
status: To Do
assignee: []
created_date: '2026-07-26 03:27'
labels:
  - ingest
  - performance
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Each ingest submission saves its options one key at a time, and every save re-reads and re-parses the entire config file and invalidates the global settings cache. A single submission triggers several full reload cycles, which is wasteful for one file and grows with every option and every submission.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Submitting an ingest writes its options in a single batched save
- [ ] #2 The global settings cache is invalidated at most once per submission
- [ ] #3 Saved option values are unchanged from the previous behaviour
<!-- AC:END -->
