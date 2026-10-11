---
id: TASK-21239
title: >-
  Kanban_Interop local store has zero consumers - retire it or justify its boot
  cost
status: Done
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

## Implementation Plan (2026-10-11 dispatch)

The owner decision this task waited on is answered by the 2026-10-11
board-audit dispatch keeping it LIVE with the burn-down's non-destructive
default: **KEEP the Kanban interop surface, defer its boot-time
construction to first use** (the task's own third AC branch). Retiring the
subsystem outright remains the owner's call; nothing here forecloses it.

1. Remove the eager trio construction from
   `_wire_watchlists_and_notifications_services`.
2. Add lazy first-use properties (`kanban_scope_service`,
   `server_kanban_service`, `local_kanban_service`) built as one locked
   trio; function-local imports so the import chain (TASK-21107) stays off
   boot.
3. Initialize the backing slots in `TldwCli.__init__`.
4. Pin: census prefix absent at `_ui_ready`; boot-DB test still green (the
   Kanban DB file is now created only on first access, not at boot).

## Implementation Notes (2026-10-11, dev `a3c47c6c72`)

- **Decision recorded**: KEEP + defer. `app_service_wiring.py`'s
  `_build_kanban_services()` builds the trio under `_kanban_services_lock`
  on first property access; the module-level `Kanban_Interop` import is
  gone (also closes TASK-21107). Consumers verified before the change:
  only tests and the Backup/Recovery participant registry (module-path
  strings, resolution untouched); no screen reads the services.
- **Measured**: `_ui_ready` census warm boot 1033 (base) -> **1022**
  (cumulative with TASK-21247's notes-sync gate); all 6 Kanban modules now
  census-pinned absent across the first-paint window.
  `tldw_chatbook_kanban.db` is no longer created at boot
  (`Tests/App/test_boot_no_feature_db_files.py` green, 2 passed).
- **No regression**: `Tests/UI/test_screen_navigation.py::
  test_app_wires_local_and_server_skills_services` (isinstance + identity
  assertions on the trio) is red at HEAD too -- the documented local
  shared-profile RecoveryRequired class (verified by checkout swap); the
  properties satisfy its assertions by construction (trio built once, so
  `kanban_scope_service.local_service is local_kanban_service` holds).
- **If the owner later retires the surface**: delete the property trio +
  slots + the `Kanban_Interop` package + `Tests/Kanban/`; the census and
  deferral pins will name anything left behind.
