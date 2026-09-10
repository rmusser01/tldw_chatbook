---
id: TASK-32180
title: >-
  Library Notes: Folder files waits are unmeasured and under-documented
status: To Do
assignee: []
created_date: '2026-09-09 09:18'
updated_date: '2026-09-09 09:18'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - file-notes
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from the Notes critique fix wave (plan
Docs/superpowers/plans/2026-09-09-library-notes-critique-wave.md); raised in
the task review of task-32055/task-32121. Three loose ends in Folder files'
folder-change waits: the slow-wait "still working" busy row has no pinned
geometry test below 120 columns; the pre-link **Use \<folder\>** button only
ever reads the legacy `notes.sync_directory` config key, not any modern
equivalent; and the invariant that every re-entrant root change abandons the
previous one synchronously (so the previous scan's lock is always released
before the new one starts) lives only in the call graph between
`_abandon_root_change_task` and its callers, with no assertion or comment
recording it for the next reader.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The slow-wait busy row's layout is pinned in a test at 60 columns
- [ ] #2 The pre-link **Use \<folder\>** button also honours the modern
  config key, or the guide explicitly names the legacy
  `notes.sync_directory` key it reads
- [ ] #3 `_abandon_root_change_task` carries an assertion or comment
  recording the synchronous-abandon invariant
<!-- AC:END -->
