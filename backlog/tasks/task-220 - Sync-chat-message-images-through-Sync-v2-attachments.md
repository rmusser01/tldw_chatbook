---
id: TASK-220
title: Sync chat message images through Sync v2 attachments
status: To Do
assignee: []
created_date: '2026-07-13 09:30'
labels:
  - sync
  - console
dependencies:
  - task-57
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console image messages (PR #621) sync text-only: the Sync v2 chat enqueue passes content fields and skips image bytes by design. Once the Sync v2 attachment upload client (task-57) is available, carry chat message images through it so a synced conversation round-trips its images. Depends on task-57.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Sending an image message enqueues its image via the Sync v2 attachment path alongside the chat message
- [ ] #2 Restore/pull rehydrates the image on the receiving side (chip renders; bytes present)
- [ ] #3 Text-only fallback preserved when the attachment API is unavailable
<!-- AC:END -->
