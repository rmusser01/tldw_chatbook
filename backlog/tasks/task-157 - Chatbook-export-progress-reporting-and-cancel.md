---
id: TASK-157
title: Chatbook export progress reporting and cancel
status: To Do
assignee: []
created_date: '2026-07-11 22:01'
labels:
  - follow-up
  - export
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Library export worker shows a static Exporting… line because ChatbookCreator has no progress callback or cancel hook. Add progress hooks to the creator and surface a progress line + cancel control in the export form. v1 deliberately shipped without this.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Export form shows real progress while a large export runs,User can cancel an in-flight export,Creator exposes progress/cancel hooks
<!-- AC:END -->
