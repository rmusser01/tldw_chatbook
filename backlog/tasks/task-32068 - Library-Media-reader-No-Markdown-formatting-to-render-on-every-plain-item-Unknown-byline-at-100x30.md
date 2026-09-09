---
id: TASK-32068
title: >-
  Library Media reader: 'No Markdown formatting to render' on every plain item;
  'Unknown' byline at 100x30
status: Done
assignee: []
created_date: '2026-09-08 18:25'
updated_date: '2026-09-08 19:23'
labels:
  - library
  - media
  - copy
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every plain-text item shows 'No Markdown formatting to render — showing the stored text' above the body, and at 100x30 an item without an author shows an 'Unknown' byline although the guide says the byline appears only when an author exists. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 19.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Markdown notice appears once in Info rather than on every render
- [x] #2 No byline is painted for items without an author at any width
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: the Markdown notice paints in Read on every plain item; an author stored as the literal 'Unknown' paints a byline at 100x30.
2. Move the notice out of the Read body into the Info body (same id, same copy) -- Read keeps the Rendered|Raw toggle for markdown items.
3. Treat the ingest placeholder author 'Unknown' as no author when building metadata_lines, so both the byline and the Info Author line drop it (edit-form prefill untouched).
4. Extend the two pinning tests in test_library_media_reader_flow.py to look in Info; GREEN; docs stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both halves are one-place presentation fixes.

AC#1: the 'No Markdown formatting to render — showing the stored text' line moved out of the Read body (`_compose_content_mode_toggle`) into the Info body. It is a fact ABOUT the item -- which is where Info lives -- and most items are plain, so as a Read banner it greeted nearly every open with a line about a view the reader never asked for. Read keeps the Rendered|Raw toggle for markdown items; an item with no content still gets no note anywhere. Same widget id and copy, so the two pinning tests in test_library_media_reader_flow.py only had to press Info first.

AC#2: an author stored as the literal 'Unknown' is now treated as no author when metadata_lines is built (library_media_viewer_state.py). That string is what audio/video/PDF/ebook ingestion writes when a file names none, so it is the ABSENCE of an author, and the byline reads those lines -- one guard drops the byline and the Info 'Author:' line together. `edit_fields` is untouched, so the edit form still prefills exactly what is stored.

Files: tldw_chatbook/Widgets/Library/library_media_viewer.py, tldw_chatbook/Library/library_media_viewer_state.py, Tests/UI/test_library_crit8_polish_media.py, Tests/UI/test_library_media_reader_flow.py (two pinning tests extended), Docs/User_Guide/library/media-and-conversations.md.
<!-- SECTION:NOTES:END -->
