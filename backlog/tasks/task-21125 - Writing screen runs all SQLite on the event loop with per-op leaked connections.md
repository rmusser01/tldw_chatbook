---
id: TASK-21125
title: >-
  Writing screen runs all SQLite on the event loop with per-op leaked connections
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - database
  - writing
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21125).

`Writing_Interop/local_writing_service.py:56-78`: ~45 `with self._connect()` sites open a fresh
connection per operation (sqlite3's `with conn:` is a transaction manager, not a closer -
GC-only leak), and the entire call chain from `UI/Writing_Window.py` / writing_controller has
zero thread offload - tree clicks and autosave run open + query + commit on the Textual event
loop, each open paying the private-seam's ~4 artifact verifications.

## Acceptance Criteria

- [ ] The service holds a thread-local connection (WAL+NORMAL already set) and closes it explicitly on shutdown
- [ ] Controller calls route through asyncio.to_thread; a thread-assert (or log probe) confirms no SQLite on the loop from this screen
- [ ] Writing screen behavior unchanged - existing tests green
