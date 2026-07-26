---
id: TASK-662
title: Support directory sources in the Library ingest job runner
status: To Do
assignee: []
created_date: '2026-07-26 03:26'
labels:
  - ingest
  - bug
  - p1
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pre-flight tells the user a folder contains N ingestible files and lets them start the job, but the runner then fails the whole thing because it treats the directory as a single file. Batch ingestion from a folder is advertised throughout the UI and does not work at all.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Submitting a folder ingests each supported file it contains
- [ ] #2 The queue reflects per-file outcomes rather than one failure for the folder
- [ ] #3 Unsupported files inside the folder fail individually without failing the supported ones
- [ ] #4 The directory scan honours the configured scan limit
<!-- AC:END -->
