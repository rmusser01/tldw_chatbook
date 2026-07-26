---
id: TASK-684.3
title: Bring web clipping into the Library ingest canvas
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
Web clipping is a distinct way to get content into the Library and is only reachable from the window being retired.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A web page can be clipped into the Library from the ingest canvas
- [ ] #2 Clipped pages land in the queue like any other import
- [ ] #3 Clipper scope and destination settings remain available
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Use TLDWAPIClient.ingest_web_content (with web_clipper_schemas.py) rather than porting WebClipperPanel's internals; ServerWebClipperService and WebClipperScopeService are already constructed in app.py and survive the window's deletion.
2. Treat a URL as what it already is in this form -- a valid ingest source -- so clipping is a behaviour of the existing path field, not a fourth mode.
3. Route clipped pages through the job registry so they land in the queue like any other import.
4. Preserve scope and destination settings.
5. Tests plus a live clip.
<!-- SECTION:PLAN:END -->
