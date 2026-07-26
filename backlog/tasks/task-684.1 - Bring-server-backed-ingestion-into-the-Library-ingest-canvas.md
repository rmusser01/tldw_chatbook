---
id: TASK-684.1
title: Bring server-backed ingestion into the Library ingest canvas
status: To Do
assignee: []
created_date: '2026-07-26 04:33'
updated_date: '2026-07-26 04:45'
labels:
  - ingest
  - consolidation
dependencies: []
parent_task_id: TASK-684
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Server Sources is the only way to start a server-backed ingest, and it lives in a window the Library canvas is meant to replace. The canvas currently tells the user ingest runs on Local, with no way to target a server.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A server-backed ingest can be started from the Library ingest canvas
- [ ] #2 The canvas shows which backend an ingest will run on and lets the user choose when both are available
- [ ] #3 The Local-only quiet line no longer appears when a server backend is configured
- [ ] #4 Starting a server ingest with no server configured explains what to configure
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reuse the existing server seam rather than extracting one. ServerMediaReadingService already wraps TLDWAPIClient.submit_media_ingest_jobs / list_media_ingest_jobs / cancel_media_ingest_jobs_batch, with tests in Tests/Media/test_server_media_ingest_jobs_service.py and Tests/tldw_api/test_media_ingest_jobs_client.py. The 700-line widget-coupled handler in tldw_api_events.py does NOT need untangling to get a server submit -- it can be left to die with the window in 671.4.
2. Map the Library ingest form to MediaIngestSubmitRequest / MediaIngestJobSubmitRequest, driven by the type group pre-flight already detects.
3. Give the ingest form a real backend choice. build_library_ingest_state takes runtime_source today and, per its own docstring, uses it for nothing but a quiet line, so the choice, its gating and its copy all belong here.
4. Route submit through the chosen backend; keep the local path exactly as it is.
5. Gate honestly: with no server configured the option explains what to configure rather than failing at submit.
6. Tests at the mapping and routing seams, then a live pass against a configured server.
<!-- SECTION:PLAN:END -->
