---
id: TASK-22224
title: >-
  Add isolation_level=None to held-connection stores, including the template
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - database
  - hardening
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22224).

Latent but load-bearing (mechanism verified empirically this review on Python 3.12/SQLite
3.49.1): without `isolation_level = None`, any bare DML on a held connection opens an
implicit DEFERRED transaction; `TransactionContextManager` then BORROWS it
(`ChaChaNotes_DB.py:18969-18978`) and `transaction(immediate=True)` silently degrades —
defeating the task-21100 hardening (12 IMMEDIATE sites) the moment one bare DML lands.
`ChaChaNotes_DB.py:3121-3141` never sets it; nor do `DB/Library_Ingest_Jobs_DB.py:48-60`
(the store TEMPLATE others copy), `DB/Evals_DB.py:180-203`,
`Widgets/Tamagotchi/tamagotchi_storage.py`, `Sync_Interop/sync_state_repository.py`,
`Kanban_Interop/local_kanban_db.py`, `Scheduling/db/scheduled_tasks_db.py`,
`Sync_Interop/notes_mirror.py`. `Notifications/event_state_repository.py:224` is the one
store that gets it right. Currently zero firing call sites were found — this is a loaded
gun, filed as hardening under the stability-over-quick-wins ruling.

## Acceptance Criteria

- [ ] All held-connection stores set `isolation_level = None` at open (or the exception is documented at the site)
- [ ] A test reproduces the degradation mechanism (bare DML then `transaction(immediate=True)` must still BEGIN IMMEDIATE) and pins it repo-wide or at the template
- [ ] The template file (`Library_Ingest_Jobs_DB.py`) carries the rule in its docstring so copies inherit it
