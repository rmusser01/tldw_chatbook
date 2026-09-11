---
id: TASK-32305
title: >-
  library_ingest_jobs.py: dismiss() docstring says 'FAILED or CANCELLED' but
  SKIPPED has been dismissible since task-2220
status: To Do
assignee: []
created_date: '2026-09-11 00:54'
labels:
  - library
  - docs
  - test-health
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The docstring on `dismiss()` in `Library/library_ingest_jobs.py` lags `_DISMISSIBLE_STATES`, which has included SKIPPED since task-2220. Qodo raised it as a High false positive on PR #2577 and will keep doing so on every PR that touches dismissal; the behaviour is pinned end-to-end on that PR, only the prose is wrong.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The dismiss() docstring names exactly the states in _DISMISSIBLE_STATES
<!-- AC:END -->
