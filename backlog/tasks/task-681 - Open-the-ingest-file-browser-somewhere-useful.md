---
id: TASK-681
title: Open the ingest file browser somewhere useful
status: Done
assignee: []
created_date: '2026-07-26 03:27'
updated_date: '2026-07-26 04:25'
labels:
  - ingest
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Browse action opens wherever the process happens to have been started, which for a first-time user is an arbitrary directory, and it lists every file regardless of whether it can be ingested. Finding the file you actually want takes more work than typing the path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The browser opens at the last used ingest location, falling back to the user's home directory
- [x] #2 The last used location survives a restart
- [x] #3 Files that cannot be ingested are visually distinguished from ones that can
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
FileOpen's location defaulted to '.', the directory the process happened to be started from, which for anyone launching from a shell has none of their documents in it. The browser now opens at the directory of the last source imported, falling back to the user's home; the location is persisted so it survives a restart. The picker also gained an 'Importable files' filter derived from the ingest capability layer, so it cannot drift from what the pipeline accepts.

Changed: tldw_chatbook/UI/Screens/library_screen.py, Tests/UI/test_library_screen.py
<!-- SECTION:NOTES:END -->
