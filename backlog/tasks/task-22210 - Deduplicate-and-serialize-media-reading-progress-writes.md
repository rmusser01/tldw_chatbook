---
id: TASK-22210
title: >-
  Deduplicate and serialize media reading-progress writes
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - library
  - database
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22210).

New with PR #2064. `_capture_library_media_loaded_progress`
(`library_screen.py:33239-33262`) fires an SQLite upsert worker on every traversal step
and every mode switch, with no `exclusive=True` and no equality skip — holding an arrow
key through 30 rows queues 30 concurrent `to_thread` writers contending for the same
write lock.

## Acceptance Criteria

- [ ] An unchanged offset produces no write; identical consecutive captures are skipped
- [ ] In-flight progress writes are superseded, not stacked (exclusive worker group or coalescing)
- [ ] A 30-row traversal produces at most one settled write (probe)
