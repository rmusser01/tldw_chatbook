---
id: TASK-3403
title: MIME-driven generated-video file extensions
status: To Do
assignee: []
created_date: '2026-08-09 04:39'
labels:
  - video
  - generation
  - storage
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct the generated-video storage boundary so validated result MIME/container data determines the stored filename extension across providers. This task is independent of workflow packaging and image-generation work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Supported generated-video MIME/container values map to a single safe canonical filename extension.
- [ ] #2 Video storage derives the filename extension from validated result metadata instead of assuming MP4.
- [ ] #3 Unknown, contradictory, or unsupported MIME/container results fail before bytes are persisted.
- [ ] #4 Existing message-name resolution, retention, eviction, tombstone, and save-copy behavior remains correct for every supported extension.
- [ ] #5 Focused validation and VideoStore tests cover MP4 and at least one non-MP4 supported container.
<!-- AC:END -->
