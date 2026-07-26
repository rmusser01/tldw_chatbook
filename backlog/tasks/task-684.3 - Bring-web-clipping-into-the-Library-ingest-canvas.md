---
id: TASK-684.3
title: Bring web clipping into the Library ingest canvas
status: In Progress
assignee:
  - '@claude'
created_date: '2026-07-26 04:33'
updated_date: '2026-07-26 14:45'
labels:
  - ingest
  - consolidation
dependencies: []
parent_task_id: TASK-684
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Web clipping is a distinct way to get content into the Library and is only reachable from the window being retired.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A web page can be clipped into the Library from the ingest canvas
- [x] #2 Clipped pages land in the queue like any other import
- [x] #3 Clipper scope and destination settings remain available
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Use TLDWAPIClient.ingest_web_content (with web_clipper_schemas.py) rather than porting WebClipperPanel's internals; ServerWebClipperService and WebClipperScopeService are already constructed in app.py and survive the window's deletion.
2. Treat a URL as what it already is in this form -- a valid ingest source -- so clipping is a behaviour of the existing path field, not a fourth mode.
3. Route clipped pages through the job registry so they land in the queue like any other import.
4. Preserve scope and destination settings.
5. Tests plus a live clip.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Web clipping now happens in the Library ingest canvas: a URL is just a source in the existing path field, not a fourth mode.

Shape: a pure mapping layer (Library/web_clip_request.py, 29 tests) turns a source + the canvas's option snapshot into the kwargs ServerMediaReadingService.ingest_web_content already takes. Kept separate from server_ingest_request rather than branching inside it -- the ingest-jobs API has no media type for a page, because clipping is a different endpoint with a different shape. Routing splits on what the source IS: a page goes to the clipper, a file to the jobs API, under one backend switch.

The local backend needed no routing at all. classify_ingest_source already calls a page an 'article' and the pipeline already extracts it via web_article_ingestion, so clipping works locally without a server or the runtime-policy gate. That was the most useful thing found here -- the task was scoped as porting a server-only feature.

Checking the endpoint's real contract BEFORE building on it (the lesson from 684.2, where a schema typed from assumption broke every completed job) found it broken for every API-key user: /api/v1/media/ingest-web-content requires a lowercase 'token' header the client never sent, so every clip 422'd before doing any work. Both headers turn out to be needed -- token for validation, X-API-KEY for auth. The trap: token alone returns 429 rate_limited, which reads as 'back off and retry' and never succeeds; it is this server's shape for a DENIAL. A control request against a route known to accept X-API-KEY is what distinguished the two.

Two facts from a real 200 shaped the design: the endpoint is SYNCHRONOUS (content returned directly, no job or batch id) so a clip has nothing to poll and settles when the call returns; and media_ids is never returned, so a finished clip cannot link to what the server made, exactly as with remote ingest jobs.

clip_failure_reason exists because a 200 is not a captured page: the outcome is in the body, so a per-result extraction_successful of False is a failed clip that looks like success at the transport level. Recording that as done would repeat task-677's empty-ingest bug. Mutation-checked.

Two prerequisites had to be fixed first, both filed and both live bugs in their own right: task-690 (a YouTube URL pre-flighted as an unsupported FILE, because grouping went by file extension while the pipeline called the same URL video) and task-697 (the pre-flight's HEAD probe turned its own 403 into 'URL unreachable' and dropped the source -- blocking a Wikipedia page that the server clipped at 200).

Scope settings are declared once in the capability schema's new 'web' group, so the form and the request cannot disagree; a test asserts every method offered is in the server's ScrapeMethod enum. Gating the page/depth limits needed OptionField.enabled_when_values, because enabled_when is a truthiness test and every non-empty select choice is truthy -- max_pages would have read as editable even for the single-page method that ignores it.

Live-verified: a real clip through the real mapping and the real service returned 200 with extraction_successful=True and 2801 characters, and clip_failure_reason correctly passed it.

Files: Library/web_clip_request.py (new), Library/ingest_capabilities.py (web group, enabled_when_values, URL-aware get_type_group), Library/ingest_preflight.py (UrlProbe), Widgets/Library/library_ingest_canvas.py (value-aware gate), app.py (clip routing + _submit_web_clip_job/_send_web_clip_job), tldw_api/client.py (token header).
<!-- SECTION:NOTES:END -->
