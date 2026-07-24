---
id: TASK-528
title: >-
  Library skills import: second submit cancels UI await but in-flight install still lands
status: To Do
assignee: []
created_date: '2026-07-24 14:10'
updated_date: '2026-07-24 14:10'
labels:
  - skills
  - library
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
All Library skill-import paths (loose file, folder, zip, URL) run the service call on a thread via the exclusive worker group; submitting a second import cancels the first path's coroutine, but the threaded service call runs to completion, so the first skill can land trust-pending silently while the UI reports only the second outcome. Pre-existing pattern; the URL path's network fetch widens the window from milliseconds to minutes (PR #831 final-review M4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Either the Import control is disabled while an import is in flight, or a cancellation check prevents the superseded install from landing after its UI await is cancelled.
- [ ] #2 Behavior is consistent across loose-file, folder, zip, and URL import paths.
- [ ] #3 A test covers the superseded-import scenario.
<!-- AC:END -->
