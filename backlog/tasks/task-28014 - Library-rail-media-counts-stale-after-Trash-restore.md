---
id: TASK-28014
title: Library rail - media counts stale after Trash restore
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
Live-reproduced: after restoring an item from Trash, the canvas title showed Media (3) while the rail row still said Media (2) and the rail Details section still said Media 2. The rail count refresh is not triggered by the restore mutation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Rail media count and Details tally match the canvas immediately after restore and other Trash mutations
- [ ] #2 A pinning test covers the restore-count path
<!-- AC:END -->
