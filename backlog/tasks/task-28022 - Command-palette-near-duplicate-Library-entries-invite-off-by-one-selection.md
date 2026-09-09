---
id: TASK-28022
title: Command palette - near-duplicate Library entries invite off-by-one selection
status: To Do
assignee: []
created_date: '2026-09-02 04:11'
labels:
  - ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Querying library yields four near-identical entries (Tab Navigation: Switch to Library; Tab Navigation: Library - Skills; Media and Content: Open Media Library; Library: Import). The 2026-09-01 live UX run's arrow-count selection landed on the wrong entry. Dedupe, reorder, or differentiate so the most common destination is the obvious first pick.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A library query presents visually distinct, deduplicated choices
- [ ] #2 The most common destination ranks first
<!-- AC:END -->
