---
id: TASK-21132
title: >-
  Note-folder managed-membership CTE anchors on the whole closure instead of the requested subtrees
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - notes
  - database
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21132).

`Notes/note_folder_repository.py:1831-1861`: the recursive CTE's anchor ignores the requested
folder_ids and filters only at the end, so every Notes-tree interaction walks the entire
managed-membership closure - twice per tree refresh (library_screen.py:11805,11943; off-loop,
so latency not freeze).

## Acceptance Criteria

- [ ] The CTE anchor is seeded from the requested ids' subtrees; results identical on existing fixtures
- [ ] A timing probe on a deep synthetic tree shows the reduction
