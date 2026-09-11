---
id: TASK-32382
title: 'Library Export: format_export_bytes is KB-only, so large libraries read as six-figure KB'
status: To Do
assignee: []
created_date: '2026-09-11 10:30'
labels:
  - library
  - export
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`format_export_bytes()` renders every size in kilobytes, so a multi-gigabyte selection estimates as "about 1048576 KB before compression". `library_ingest_state._human_size` already steps units and is the obvious model. The formatter is deliberately shared by the pre-export estimate and the post-export receipt so the two can never round differently, which sets the constraint on the fix: both halves move together, in one change, re-pinning the critique-9 receipt string at the same time. Recorded as a `ponytail:` comment at the formatter during the task-32353 review round.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A size above 1 MB renders in a unit a person reads without counting digits, in both the estimate and the receipt
- [ ] #2 The estimate and the receipt still round through one shared function
- [ ] #3 The critique-9 receipt-string pin is updated in the same change rather than left asserting the old formatting
<!-- AC:END -->
