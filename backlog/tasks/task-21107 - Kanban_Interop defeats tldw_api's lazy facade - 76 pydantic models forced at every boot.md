---
id: TASK-21107
title: >-
  Kanban_Interop defeats tldw_api's lazy facade - 76 pydantic models forced at every boot
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - startup
  - imports
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21107).

`Kanban_Interop/server_kanban_service.py:10` does a module-level 31-name `from ..tldw_api
import`, forcing `tldw_api/kanban_schemas.py` (76 pydantic models, ~44 ms self) through the
otherwise-lazy PEP-562 facade - one of exactly two leaks (the other is fixed by TASK-21106).

## Acceptance Criteria

- [ ] The import is TYPE_CHECKING/function-local; kanban behavior (such as it is - zero UI consumers found) unchanged
- [ ] A test asserts `tldw_chatbook.tldw_api.kanban_schemas` is not in sys.modules after importing the app module
