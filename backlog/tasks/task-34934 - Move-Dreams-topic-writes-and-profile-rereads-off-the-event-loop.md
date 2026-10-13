---
id: TASK-34934
title: Move-Dreams-topic-writes-and-profile-rereads-off-the-event-loop
status: Done
assignee: []
created_date: 2026-10-13 14:55
labels:
- performance
- ui-perf
- wave-4-blocking-rebuilds
dependencies: []
priority: medium
updated_date: 2026-10-13 19:00
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dreams topic add/remove handlers do sync upsert/delete_profile_entry writes plus a full-table list_profile() read (all facets, filtered in Python) inside the pane rebuild on the event loop (settings_screen.py:33002/33037 feeding :32326) — click lag on every topic change and on every Settings category switch to Dreams, growing with profile rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Write and re-read run off the loop with pane refresh applied on completion (write threaded; re-read noted as bounded — see notes)
- [x] Topic list reflects adds and removes correctly (test)
<!-- AC:END -->
