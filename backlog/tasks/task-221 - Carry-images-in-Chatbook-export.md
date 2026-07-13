---
id: TASK-221
title: Carry images in Chatbook export
status: To Do
assignee: []
created_date: '2026-07-13 09:30'
labels:
  - chatbooks
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Save Chatbook currently omits message images: a Chatbook exported from a Console conversation with image messages (PR #621) loses them. Extend the Chatbook schema/packaging to carry image attachments and restore them on import (adjacent to task-19's attachment-availability work).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Exporting a conversation with image messages includes the image bytes in the Chatbook
- [ ] #2 Importing that Chatbook restores messages with working chips and Save Image
- [ ] #3 Chatbook schema/version bump documented
<!-- AC:END -->
