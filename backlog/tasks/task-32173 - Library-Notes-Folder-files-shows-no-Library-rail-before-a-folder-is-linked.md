---
id: TASK-32173
title: >-
  Library Notes: Folder files shows no Library rail before a folder is
  linked
status: To Do
assignee: []
created_date: '2026-09-09 09:11'
updated_date: '2026-09-09 09:11'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - file-notes
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from the Notes critique fix wave (plan
Docs/superpowers/plans/2026-09-09-library-notes-critique-wave.md); raised in
the task review of task-32136. task-32136 AC#1 shipped the Library rail and
the source strip staying visible inside Folder files once a folder is
linked, but only partially: `#file-notes-body` is hidden entirely while
`_root is None`, so the pre-link empty state still drops to a full-width
layout with only the source strip surviving — the same gap the file-notes
guide's own pre-link sentence already documents as a known limitation
rather than a fixed behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At wide sizes, the Library rail is visible in the empty (no root
  linked) state, not just after a folder is linked
- [ ] #2 Compact-terminal behavior is unchanged
- [ ] #3 The file-notes guide's pre-link sentence is updated to match
- [ ] #4 The behavior is pinned in a test
<!-- AC:END -->
