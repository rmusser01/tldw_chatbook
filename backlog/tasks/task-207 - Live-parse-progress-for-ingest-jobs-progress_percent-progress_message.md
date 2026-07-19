---
id: TASK-207
title: Live parse progress for ingest jobs (progress_percent/progress_message)
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
Ingest jobs show only a coarse state (parsing/writing). Borrow the server jobs schema's progress_percent/progress_message idea: let the parse worker report incremental progress (e.g. per-file %, or a stage message like 'transcribing 3/5') surfaced on the job row, without adding new job states. Discovered while mining tldw_server2 core Jobs module during task 161. Requires a progress field on LibraryIngestJob + a worker→UI progress channel (call_from_thread) + persistence if desired.
<!-- SECTION:DESCRIPTION:END -->
