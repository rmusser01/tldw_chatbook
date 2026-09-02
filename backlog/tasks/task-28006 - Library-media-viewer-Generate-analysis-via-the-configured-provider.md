---
id: TASK-28006
title: Library media viewer - Generate analysis via the configured provider
status: To Do
assignee: []
created_date: '2026-09-02 04:10'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The only analysis-generation trigger in the product is the ingest-time Analyze-after-import checkbox (default OFF, inside a collapsed panel). The viewer Analysis section is explicitly edit-only (Add analysis opens an empty TextArea; generation out of scope per library_media_viewer.py:404-409 docstring), and the old generation workbench (Widgets/Media/media_viewer_panel.py) is stranded on the retired Media screen route. A user who imported without analysis has no recovery except re-import with Overwrite. Wire a Generate analysis action into the viewer Analysis section: resolve the provider through the existing resolve_ingest_analysis_provider seam (Library/ingest_analysis.py) and persist through the existing save_analysis_version service, reusing the same promise/receipt honesty (an unready provider shows the reason).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An item without an analysis offers a Generate action when an analysis provider is ready
- [ ] #2 Generation persists the result as an analysis version visible in the viewer
- [ ] #3 With no ready provider, the action communicates the same reason language as the ingest hint instead of silently failing
- [ ] #4 Generation runs off the UI thread with visible progress
<!-- AC:END -->
