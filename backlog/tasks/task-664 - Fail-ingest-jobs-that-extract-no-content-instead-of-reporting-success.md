---
id: TASK-664
title: Fail ingest jobs that extract no content instead of reporting success
status: To Do
assignee: []
created_date: '2026-07-26 03:26'
labels:
  - ingest
  - bug
  - p1
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A PDF ingested without usable extraction is recorded as done, counted in the library media total, and given a media row whose content is empty. The user has no signal that the import produced nothing, and the entry silently returns no results from search and RAG.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An ingest that extracts no content is reported as failed, not done
- [ ] #2 The failure names the file and explains that no content could be extracted
- [ ] #3 No empty-content media row is left behind in the library
- [ ] #4 Files that legitimately contain no text are distinguished from extraction failures in the message
<!-- AC:END -->
