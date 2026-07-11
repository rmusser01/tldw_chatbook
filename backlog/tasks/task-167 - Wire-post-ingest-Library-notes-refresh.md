---
id: TASK-167
title: Wire post-ingest Library notes refresh
status: To Do
assignee: []
created_date: '2026-07-11 22:02'
labels:
  - follow-up
  - library
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The removed refresh_notes_tab_after_ingest call was already a no-op since Notes consolidated into Library, so Ingest-tab note imports have never refreshed the Library notes list. Wire a real refresh so a note imported via ingest appears in the Library notes canvas without a manual reload.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A note imported via ingest appears in the Library notes list without a manual refresh
<!-- AC:END -->
