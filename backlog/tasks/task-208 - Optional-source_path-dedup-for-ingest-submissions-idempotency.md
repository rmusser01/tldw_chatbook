---
id: TASK-208
title: Optional source_path dedup for ingest submissions (idempotency)
status: To Do
assignee: []
created_date: '2026-07-12 17:34'
labels:
  - follow-up
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The ingest registry allows submitting the same source_path multiple times (N duplicate jobs). Borrow the server jobs schema's idempotency_key/partial-unique-index idea as an OPTIONAL guard: warn/skip when a source_path is already queued or in-flight (NOT a hard unique constraint — re-ingesting a changed file is legitimate). Design decision needed: dedup scope (in-flight only vs all-history), and opt-in vs default. Discovered while mining tldw_server2 core Jobs module during task 161.
<!-- SECTION:DESCRIPTION:END -->
