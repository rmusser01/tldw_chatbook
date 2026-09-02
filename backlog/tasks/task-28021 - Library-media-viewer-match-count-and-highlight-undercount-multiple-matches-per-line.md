---
id: TASK-28021
title: >-
  Library media viewer - match count and highlight undercount multiple matches
  per line
status: To Do
assignee: []
created_date: '2026-09-02 04:11'
labels:
  - library
  - bug
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
find_content_matches reports a line at most once and the raw-mode highlighter styles only the first occurrence per line (library_media_viewer_state.py:377-381; library_media_content.py:43-53), so Match 3 of 7 undercounts dense lines; rendered mode gets no highlights at all (sync_search targets the raw widget only).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The match count reflects all occurrences
- [ ] #2 All occurrences on a line are highlighted in raw mode
- [ ] #3 Rendered mode either highlights matches or states why it cannot
<!-- AC:END -->
