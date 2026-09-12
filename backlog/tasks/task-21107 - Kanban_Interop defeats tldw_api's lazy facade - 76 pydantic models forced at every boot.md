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
priority: low
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

## Re-verification against dev 2be18842a (2026-08-23)

An independent read-only pass re-checked this finding before implementation. **Still true, but
the cost is over-billed ~2x and the acceptance criteria as written cannot be met.**

**Confirmed**: `Kanban_Interop/server_kanban_service.py:10-42` is a module-level import of 31
names from `..tldw_api`, reached from `app.py:542` via `Kanban_Interop/__init__.py:5` →
`kanban_scope_service.py:9`. The 76 pydantic classes in `kanban_schemas.py` is exact. It is a
*sole* leak — `tldw_api.client` is not in the eager boot closure, so nothing else drags it in.

**Cost corrected**: ~19 ms warm, not ~44 ms. The original number was measured with pydantic's
own internals cold and charged to this module; 19 of the boot closure's modules already import
pydantic, so that cost is paid regardless.

**Why the prescribed fix does not work**: `KANBAN_OPERATION_SPECS` stores the schema *classes*
as runtime values (`request_model=KanbanBoardCreate`), and two siblings read that dict at their
own module scope (`local_kanban_service.py:29-32`, `kanban_scope_service.py:43`). A
TYPE_CHECKING or function-local import breaks the import outright. There is already a test
documenting this as deliberate — `Tests/Utils/test_tldw_api_schema_deferral.py:40-48` allowlists
`kanban_schemas` as "a genuine module-scope need, not an oversight".

**Revised direction**: store the model *name* in the spec table and resolve it lazily in
`_coerce_request_args` (`server_kanban_service.py:702-723`, the only runtime reader). The house
pattern already exists — `local_kanban_service.py:336,430,558` do function-local
`from ..tldw_api import Kanban...`. This also requires updating the allowlist test. That is a
larger and riskier change than the original filing implies; treat it as such, and if it does not
justify ~19 ms for a feature with no UI consumers, close this instead.

## Closure recommendation (2026-08-24, burn-down close-out)

**Recommend closing without work.** ~19 ms warm, not ~44 ms, and the prescribed fix cannot compile: KANBAN_OPERATION_SPECS stores the schema classes as runtime values, two siblings read that dict at module scope, and Tests/Utils/test_tldw_api_schema_deferral.py already allowlists the import as "a genuine module-scope need, not an oversight". A real fix means a lazy spec table plus editing that allowlist test -- a larger, riskier change than ~19 ms for a feature with no UI consumers justifies.

Left open rather than closed unilaterally: retiring a filed finding is the owner's call. The
evidence above is what a re-verification pass measured against dev before dispatch; if it is
accepted, close this as "retired on evidence" rather than "won't fix", because the mechanism was
real and only the cost or the prescribed fix was wrong.

