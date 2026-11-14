---
id: TASK-21107
title: >-
  Kanban_Interop defeats tldw_api's lazy facade - 76 pydantic models forced at every boot
status: Done
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


## Implementation Plan (2026-10-11 dispatch)

The board audit kept this LIVE past the 2026-08-24 closure recommendation,
so the fix ships via the revised direction's cheaper neighbour: instead of
converting `KANBAN_OPERATION_SPECS` to name-based lazy resolution (risky,
touches the spec table plus its two module-scope readers), TASK-21239's
keep-path makes the whole Kanban service trio lazy -- and the module-level
import in `app_service_wiring.py` that dragged `server_kanban_service.py`
(and through it the 31-name `..tldw_api` import) onto the app import path
goes function-local. The spec table stays untouched.

1. Remove the module-scope `Kanban_Interop` import from
   `app_service_wiring.py`.
2. Defer the trio's construction to first-use properties (see TASK-21239).
3. Update `Tests/Utils/test_tldw_api_schema_deferral.py`: drop the
   `kanban_schemas` allowlist entry and add the AC's named assertion.
4. Pin absence at `_ui_ready` too (census), since construction no longer
   happens at boot.

## Implementation Notes (2026-10-11, dev `a3c47c6c72`)

- **Fix**: with the trio lazy (TASK-21239), `import tldw_chatbook.app`
  loads ZERO Kanban modules (was 6, including
  `tldw_chatbook.tldw_api.kanban_schemas` and its 76 pydantic models). The
  spec-table shape `server_kanban_service.py` needs is untouched -- the
  module simply is not imported until a consumer touches the property.
- **Tests**: `test_kanban_schemas_absent_from_app_import_closure` (fresh
  subprocess; fails naming the offender if either `kanban_schemas` or any
  `Kanban_Interop` module re-eagers) + the phase-2 allowlist shrunk to the
  package facade (+ a reviewed `tldw_api.exceptions` entry for a
  PRE-EXISTING, unrelated leak from
  `Audio_Services_Interop/server_audio_services_service.py:11` -- that
  module uses its exception types in runtime `except` clauses across a
  dozen methods; the module is stdlib-only exception classes, so deferral
  buys nothing; the allowlist test was already red at HEAD for this
  reason, verified by checkout swap). Census
  `ABSENT_AT_READY_PREFIXES`/`ABSENT_AT_READY_MODULES` pin the package and
  `kanban_schemas` absent across the whole first-paint window. All green:
  23 passed across the census + deferral + gate files.
- **Cost note**: the 2026-08 re-verification billed this ~19 ms warm. The
  win shipped here is that plus six modules off the `_ui_ready` census
  (1033 -> 1022 cumulative with TASK-21247), achieved WITHOUT the risky
  spec-table rewrite the original filing prescribed.
