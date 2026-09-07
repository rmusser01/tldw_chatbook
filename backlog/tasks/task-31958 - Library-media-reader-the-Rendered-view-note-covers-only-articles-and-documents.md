---
id: TASK-31958
title: >-
  Library media reader - the Rendered-view note covers only articles and
  documents
status: To Do
assignee: []
created_date: '2026-09-07 08:26'
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
- [ ] #1 Any non-Markdown item with content that cannot render explains why, whatever its media type
- [ ] #2 The note does not paint above an item with no stored content
- [ ] #3 Painted pins cover a plaintext or media item and the empty-content case
<!-- AC:END -->
