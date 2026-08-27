---
id: TASK-23023
title: >-
  One import statement costs 48 ms and 8 boot modules for a single integer constant
status: To Do
assignee: []
created_date: '2026-08-27'
labels:
  - performance
  - startup
  - regression
priority: high
---

## Description

The boot import closure is **657 modules against a 660 budget**, and **15 of the 20 new modules come
from one import statement**.

`Library/library_ingest_jobs.py:77` imports a stdlib-only validator *through* the package path, so
`Research_Workspace/__init__.py:22-39` executes and eagerly re-exports the whole tree - including
`server_adapter`, which imports a 26-model pydantic module (782 LOC) for **one integer constant**,
`MAX_WORKSPACE_SOURCE_OWNER_ROWS = 10_100`.

Same class as the previously-fixed 21102/21107 facade leaks. The route is ungated: `research_workspace`
is a live shell destination, so a user who never opens it pays this on every boot.

## Acceptance Criteria

- [ ] `import tldw_chatbook.app` no longer executes `Research_Workspace/__init__`'s eager re-exports
- [ ] The boot closure drops by ~8 modules and the import cost by ~48 ms, measured with interleaved arms
- [ ] `server_adapter` no longer imports a pydantic schema module to read one integer
- [ ] The Research Workspace feature still works; a test drives it from the deferred state
- [ ] The 660 budget is **not** raised

## Evidence

Chain: `app.py:224 -> Library/server_ingest_reconcile.py:24 -> library_ingest_jobs.py:77 -> from
tldw_chatbook.Research_Workspace.source_operations import validate_source_operation_id`.

Measured, interleaved x3 pairs (arm B = package `__init__` emptied in a scratch copy):

| | tip | arm B |
|---|---|---|
| cost of that one import statement | **65.9 / 54.9 / 55.1 ms** | **7.1 / 5.5 / 6.9 ms** |
| own-module closure | 657 | **649** |
| `tldw_api.notes_workspace_schemas` resident | yes (**20.6 ms self**) | no |

`source_operations.py` itself is stdlib-only. The fix is the package-init eagerness, not the
dependency.

Source: `Docs/Design/2026-08-27-holistic-perf-review.md`.
