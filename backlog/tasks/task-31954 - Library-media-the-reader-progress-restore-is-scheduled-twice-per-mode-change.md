---
id: TASK-31954
title: Library media - the reader progress restore is scheduled twice per mode change
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
H2 review: handle_library_media_reader_mode schedules _restore_library_media_loaded_progress through the viewer seam and again through _sync_library_media_viewer_state's own call_after_refresh, unconditionally re-armed by nulling _library_media_progress_restored_id. Harmless today because scroll_to is idempotent and pre-existing, but two owners for one restore is how a future scroll change becomes a visible jump.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One owner schedules the progress restore for a reader mode change
- [ ] #2 A pin counts the restore invocations for a single mode change
<!-- AC:END -->
