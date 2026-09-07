---
id: TASK-31958
title: >-
  Library media reader - the Rendered-view note covers only articles and
  documents
status: Done
assignee: []
created_date: '2026-09-07 08:26'
updated_date: '2026-09-07 20:20'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
J Task 1 review M4/M5: the 'Rendered view is for Markdown and transcripts' note is gated on RENDERED_VIEW_NOTE_TYPES (article, document), so a plaintext, video or audio item that fails the content sniff still gets the silent blank mode-strip slot with no explanation. The note also paints above an empty article ('No stored content.'), where it explains nothing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Any non-Markdown item with content that cannot render explains why, whatever its media type
- [x] #2 The note does not paint above an item with no stored content
- [x] #3 Painted pins cover a plaintext or media item and the empty-content case
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Painted pins: a plaintext or media item whose content fails the sniff shows the note; an item with no stored content shows `No stored content.` and no note. 2. Gate the note on "has content and cannot render" instead of the article/document allowlist.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The `Rendered view is for Markdown and transcripts` note in the mode-strip slot is gated on `has_content` (the item has stored content) and the content sniff failing — the `RENDERED_VIEW_NOTE_TYPES` allowlist is gone, so a plaintext, video or audio item that cannot render explains why, and an empty item paints only `No stored content.`. Copy changed at review to type-neutral `No Markdown formatting to render — showing the stored text`, because the old wording contradicted itself above a plain-prose video/audio item. Text-only; three painted pins (RED first: NoMatches / note present; the AC#1 pin keeps a literal so a silent copy change fails). User Guide sentence corrected.
<!-- SECTION:NOTES:END -->
