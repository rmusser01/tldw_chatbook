---
id: TASK-21101
title: >-
  NotesDeviceStateStore - adopt the WAL+NORMAL held-connection template and stop re-running the schema census per transaction
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - database
  - notes-sync
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21101).

`Notes/notes_device_state_store.py:443-472` is the app's only remaining DELETE-journal +
synchronous=FULL store (live pragma read-back confirmed: journal_mode=delete, synchronous=2).
It opens a fresh connection per operation, and `initialize_notes_device_schema` re-runs a full
`sqlite_schema` census plus 16 `CREATE INDEX IF NOT EXISTS` re-executions (~60 statements)
inside EVERY transaction, including pure reads. It sits behind the notes-sync runtime (boots
unconditionally) and the notes import executor (2-4 receipt transactions per imported note: a
500-note import pays 1,000+ open/census/fsync cycles). This store dodged the task-15465/15466
sweep because it lives outside `DB/`.

## Acceptance Criteria

- [ ] The store uses the sanctioned template: held thread-local connection, WAL journal_mode, synchronous=NORMAL, isolation_level=None (exemplar `Library_Ingest_Jobs_DB.py`), with pragmas verified by read-back in a test
- [ ] `initialize_notes_device_schema` runs once per connection lifetime (the `initialize()` seam), not per transaction; a statement-count probe demonstrates the reduction
- [ ] Write transactions keep their current BEGIN IMMEDIATE semantics; existing notes-sync and import tests stay green
