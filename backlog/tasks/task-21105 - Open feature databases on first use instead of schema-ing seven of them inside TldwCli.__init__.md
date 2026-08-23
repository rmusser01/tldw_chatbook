---
id: TASK-21105
title: >-
  Open feature databases on first use instead of schema-ing seven of them inside TldwCli.__init__
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - startup
  - database
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21105).

Seven feature databases are created and schema'd synchronously inside `TldwCli.__init__` for
features a user may never touch: research (5 tables + migrations, app.py:6948), notifications
(app.py:7068), event_state (10 DDL) + sync_state (16 DDL) server-parity stores (app.py:7081),
writing (16 DDL, app.py:6723), kanban (24 DDL, app.py:7277 - zero UI consumers found at all),
notes_sync_state (start path). Each is file create + WAL setup + executescript + fsync traffic,
serial, pre-paint. The lazy seam already exists: `BaseDB.__init__(initialize_schema=False)`
(DB/base_db.py:43).

## Acceptance Criteria

- [ ] Each of the listed stores opens (and creates its schema) on first feature use, not during app construction; feature behavior on first use is unchanged
- [ ] Per-store regression tests assert no DB file exists after a boot that never touches the feature
- [ ] Boot construction time before/after recorded in the task (isolated-profile probe)
