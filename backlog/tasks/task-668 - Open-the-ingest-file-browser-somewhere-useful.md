---
id: TASK-668
title: Open the ingest file browser somewhere useful
status: To Do
assignee: []
created_date: '2026-07-26 03:27'
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
- [ ] #1 The browser opens at the last used ingest location, falling back to the user's home directory
- [ ] #2 The last used location survives a restart
- [ ] #3 Files that cannot be ingested are visually distinguished from ones that can
<!-- AC:END -->
