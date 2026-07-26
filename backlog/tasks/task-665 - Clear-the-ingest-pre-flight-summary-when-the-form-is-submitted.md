---
id: TASK-665
title: Clear the ingest pre-flight summary when the form is submitted
status: To Do
assignee: []
created_date: '2026-07-26 03:26'
labels:
  - ingest
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After a successful submit the path field empties and the form says a path is needed, while the pre-flight block still describes the file that was just submitted. The screen shows two contradictory states at once and the stale summary suggests a file is still staged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Submitting clears the pre-flight summary along with the path
- [ ] #2 The gate line and the summary never describe different states at the same time
<!-- AC:END -->
