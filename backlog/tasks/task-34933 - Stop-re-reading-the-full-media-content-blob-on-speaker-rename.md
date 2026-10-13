---
id: TASK-34933
title: Stop-re-reading-the-full-media-content-blob-on-speaker-rename
status: Done
assignee: []
created_date: 2026-10-13 14:55
labels:
- performance
- ui-perf
- wave-4-blocking-rebuilds
dependencies: []
priority: medium
updated_date: 2026-10-13 18:58
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The LibraryMediaViewer.SpeakerRenamed handler calls db.get_media_by_id(media_id), which selects every column including the whole content transcript, only to refresh detail state with row[content] (library_screen.py:6607). A large-transcript media item lags the UI right after a speaker rename on the UI thread.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Rename handler refreshes only what changed (targeted read or in-place patch of detail state)
- [x] No full-content SELECT on the rename path (test)
<!-- AC:END -->
