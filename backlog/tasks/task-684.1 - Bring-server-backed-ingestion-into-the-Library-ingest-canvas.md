---
id: TASK-684.1
title: Bring server-backed ingestion into the Library ingest canvas
status: In Progress
assignee: []
created_date: '2026-07-26 04:33'
updated_date: '2026-07-26 05:09'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Slice 1 of 3 landed: the pure mapping layer (tldw_chatbook/Library/server_ingest_request.py, 21 tests).

Key finding that reshaped the plan: the server seam already existed. ServerMediaReadingService.submit_ingest_jobs already wraps TLDWAPIClient.submit_media_ingest_jobs and accepts exactly the kwargs a Library submission needs, with list/cancel siblings for 684.2 and ingest_web_content for 684.3. The 700-line widget-coupled handler in tldw_api_events.py never needs untangling -- it dies with the window in 684.4.

Two boundaries are explicit rather than guessed: the server has no html media type (its document extractor takes that text), and a plain web page is refused with the reason that clipping runs through a different endpoint (684.3), rather than being sent as a type the server would reject.

Remaining: slice 2 routes submit through the chosen backend (the selector already exists as app.media_runtime_state.runtime_backend, alongside the local/server MediaReadingScopeService pattern); slice 3 renders the choice in the canvas and replaces the cosmetic 'ingest runs on Local' line, gating honestly when no server is configured. Then a live pass against a configured server -- which is the part this environment cannot verify.
<!-- SECTION:NOTES:END -->
