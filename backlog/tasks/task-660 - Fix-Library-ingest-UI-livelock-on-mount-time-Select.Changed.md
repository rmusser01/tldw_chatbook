---
id: TASK-660
title: Fix Library ingest UI livelock on mount-time Select.Changed
status: To Do
assignee: []
created_date: '2026-07-26 03:26'
labels:
  - ingest
  - bug
  - p0
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pointing the Library ingest path field at a PDF, audio, video or e-book file freezes the whole application at 100% CPU with no recovery, because the per-type options panel recomposes itself in an endless cycle. These are the app's primary ingest types, so the ingest screen is effectively unusable for anything but plain text.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Entering a PDF path and submitting completes without the UI freezing
- [ ] #2 The same holds for audio, video and e-book paths, and for a folder containing them
- [ ] #3 A regression test drives a type group whose options include a select field and asserts the recompose count stays bounded
- [ ] #4 Changing a per-type select value by hand still updates the panel title and dependent-field enabled states
<!-- AC:END -->
