---
id: TASK-21239
title: >-
  Kanban_Interop local store has zero consumers - retire it or justify its boot
  cost
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - cleanup
  - startup
  - technical-debt
  - needs-owner
dependencies: []
priority: low
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; evidence gathered
while implementing TASK-21105 and recorded in that task file. Split out from the TASK-21105
review's other finding (filed as TASK-21238) because retiring a subsystem is an owner
decision, not a performance fix.

While making the seven feature databases lazy, TASK-21105 established that the Kanban store
has **zero consumers** — nothing in the app reads it. On dev `b2b1e2e0d` `TldwCli.__init__`
still constructs `ServerKanbanService`, `LocalKanbanService` (pointed at
`tldw_chatbook_kanban.db` in the user data dir) and `KanbanScopeService`
(`app.py:7511-7526`), and no screen consumes any of them. Separately, TASK-21107 (still To Do)
records that `Kanban_Interop` defeats `tldw_api`'s lazy facade and forces **76 pydantic
models** at every boot.

So the app creates a database file in the user's data directory, and pays a measured import
cost, for a feature with no reachable surface.

## Acceptance Criteria

- [ ] An owner decision is recorded — retire the Kanban interop surface, or keep it with the intended consumer named
- [ ] If retired, `TldwCli.__init__` constructs no Kanban service, no Kanban database file is created in the user data dir, and the modules are removed
- [ ] If kept, its boot-time construction is deferred to first use like the other six stores, so an unused subsystem costs nothing at boot
- [ ] Either outcome leaves the `import tldw_chatbook.app` closure no larger than it is today
