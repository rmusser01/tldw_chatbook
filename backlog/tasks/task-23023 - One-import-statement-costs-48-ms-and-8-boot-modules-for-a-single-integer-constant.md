---
id: TASK-23023
title: >-
  One import statement costs 48 ms and 8 boot modules for a single integer constant
status: Done
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

- [x] `import tldw_chatbook.app` no longer executes `Research_Workspace/__init__`'s eager re-exports
- [x] The boot closure drops by ~8 modules and the import cost by ~48 ms, measured with interleaved arms
- [x] `server_adapter` no longer imports a pydantic schema module to read one integer
- [x] The Research Workspace feature still works; a test drives it from the deferred state
- [x] The 660 budget is **not** raised
- [x] A deferred Research_Workspace import that raises at first navigation surfaces legibly
      (user-visible failure, app and current screen survive, other routes keep working) --
      lazification moves failures from boot to first use, and this class has broken the app before
- [x] The deferral is proven at `_ui_ready` too, not only after import (the TASK-21731 lesson):
      the screen-only members and the pydantic schema module are absent when the app becomes usable

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

## Implementation Plan

1. Re-verify the finding on the branch base (`d7bb844d9b`): trace the chain, measure the statement
   cost and the closure with interleaved arms (arm A = base files, arm B = fix), and name every
   dropped module rather than trusting the reported 8.
2. Make `Research_Workspace/__init__` a PEP 562 lazy facade (the `tldw_api`/`Local_Ingestion`
   house pattern): `__all__` unchanged, a flat name -> submodule map, module `__getattr__` that
   imports one submodule per ask and caches, `__dir__` for tooling. No importer rewrites needed --
   `from tldw_chatbook.Research_Workspace import X` keeps working and a later direct submodule
   import stays cheap.
3. Lift `MAX_WORKSPACE_SOURCE_ROWS`/`MAX_WORKSPACE_SOURCE_OWNER_ROWS` into a stdlib-only
   `tldw_api/notes_workspace_limits.py` (the `chunking_engine_version.py`/`search_modes.py`
   pattern); `notes_workspace_schemas` re-imports them so there is one object per bound;
   `server_adapter` reads the light module.
4. Prove gone at `_ui_ready`, not relocated (the TASK-21731 lesson): census both arms at ready,
   pin the truly-absent members in `test_ui_ready_module_census.py`, and record which members
   legitimately move to the construct leg (`_wire_research_source_association`'s readiness
   adapters) in `test_construct_runtime_imports.py`'s reviewed allowlist.
5. Walk the first-use failure modes of the deferral; make the silent branch
   (`_complete_screen_navigation`'s load-failure else) legible; cover it with a broken-submodule
   subprocess test.
6. New guards in `Tests/Packaging/test_research_workspace_import_closure.py`; mutation-test every
   new test against a deliberately broken implementation; A/B every red suite against base.

## Implementation Notes

**What shipped.**

1. `tldw_chatbook/Research_Workspace/__init__.py` -- PEP 562 lazy facade. All 31 re-exported
   names resolve on first attribute access via `_SUBMODULE_BY_NAME` + module `__getattr__`
   (identical objects, cached after first ask); importing the bare package resolves zero
   submodules. The boot chain (`library_ingest_jobs` -> `source_operations`) now pays only the
   package init + the stdlib-only validator.
2. `tldw_chatbook/tldw_api/notes_workspace_limits.py` (new, stdlib-only) owns
   `MAX_WORKSPACE_SOURCE_ROWS` and `MAX_WORKSPACE_SOURCE_OWNER_ROWS`; `notes_workspace_schemas`
   re-imports both (one object per bound, no copy to drift) and `server_adapter` now imports the
   light module -- the 26-model pydantic module is severed from the adapter entirely.
3. `app.py` `_complete_screen_navigation`: the no-screen-class else branch now surfaces the
   failure (`_notify_navigation_failure` toast + nav-bar rollback) and logs the blocking
   exception via `screen_load_error()`. Rationale: lazification moves a broken submodule's
   failure from boot (dead app) to first navigation, and that branch was the one silent path
   left -- the exact task-2720 defect shape. AC added for this before implementing.
4. Census guard (`test_ui_ready_module_census.py`): controller/layout_state/overlay_store/
   notes_workspace_schemas added to `ABSENT_AT_READY_MODULES`. Construct guard
   (`test_construct_runtime_imports.py`): local_adapter/server_adapter/quick_notes/
   tldw_api.exceptions/notes_workspace_limits added as reviewed rows --
   `_wire_research_source_association` (construct-time, pre-existing) genuinely builds the
   readiness adapters, so those members move from the import phase to the construct phase
   (both pre-paint) rather than disappearing; restructuring that wiring was deliberately NOT
   done (stability-over-quick-wins).

**Measured, arms interleaved x3 (arm A = base `d7bb844d9b` files, arm B = fix; isolated
config; venv Python 3.12.11).**

| | arm A (base) | arm B (fixed) |
|---|---|---|
| the one statement, standalone (`from ...source_operations import validate_source_operation_id`) | 286.1 / 273.6 / 273.5 ms, 143 own modules | **77.6 / 79.9 / 81.3 ms, 4 own modules** |
| `import tldw_chatbook.app` own modules | 658 / 658 / 658 | **650 / 650 / 650** |
| `notes_workspace_schemas` in boot closure | yes | **no** |
| own modules at `_ui_ready` (warm 2nd boot) | 958 | **955** |

The 8 dropped boot modules, named (`comm` over sorted censuses): `Research_Workspace.
{controller, layout_state, overlay_store, local_adapter, server_adapter, quick_notes}`,
`tldw_api.exceptions`, `tldw_api.notes_workspace_schemas`; nothing added in arm B (the limits
module is not on the boot path at all). At `_ui_ready`: controller/layout_state/overlay_store/
notes_workspace_schemas GONE; local_adapter/server_adapter/quick_notes resident via the
construct-leg readiness wiring (now without the pydantic payload); `notes_workspace_limits`
(21 lines, stdlib) is the only addition. Absolute ms differ from the tracer numbers in the
finding (different probe: standalone statement in a cold interpreter vs marginal cost inside
the app import; machine under load) -- the A/B deltas are the evidence.

**New tests** (`Tests/Packaging/test_research_workspace_import_closure.py`, 5): app-import
closure guard (+ validator anti-vacuity), facade contract (bare package pulls zero submodules;
all 31 names resolve identically; map/`__all__` agreement; AttributeError on unknown),
server_adapter limits contract (schema module absent + single-sourced bounds), deferred-state
screen drive (real headless boot to `_ui_ready`, deferred members proven absent at ready,
navigate to `research_workspace`, screen mounts with a real controller), and broken-submodule
legibility (meta-path ImportError on `controller`: app boots, first navigation shows the
"Couldn't open" notification, current screen survives, Library still opens).

**Mutation results** (each new test red against a deliberately broken implementation):
eager `__init__` restored -> 4/5 fail (closure, facade, deferred-state, legibility -- the
last because a broken submodule again kills boot); `server_adapter` re-pointed at
`notes_workspace_schemas` -> limits contract AND deferred-state fail (schemas resident at
ready via the construct leg) while the import-closure test stays green -- exactly the blind
spot the at-ready assertions exist for; `_SUBMODULE_BY_NAME` mis-mapped -> facade contract
fails; `_notify_navigation_failure` call removed -> legibility test fails.

**Test A/B (identical commands, both arms).** Branch-relevant sweep
(`Tests/Research_Workspace Tests/Packaging Tests/Library Tests/tldw_api(workspace/notes)
Tests/Utils/test_tldw_api_lazy Tests/App(construct/submit/retry)`): 3342 passed / 10 failed.
9 of the 10 fail byte-identically with the base production files restored
(`test_rag_boot_import_closure::test_chat_screen...` and `test_ui_ready_module_census` both
red for exactly `Chat.trajectory_export` -- TASK-23020's in-flight family, same single member
on both arms; `test_library_rag_state` x1 and `test_submit_library_ingest_job` x7 are
pre-existing dev reds). The 10th (`test_construct_runtime_imports`) was mine: the guard doing
its job on the construct-leg move; resolved with the reviewed allowlist rows above. UI
navigation + research-screen suites and `test_app_import_weight.py` (budget untouched at 660)
green.

**Modified/added files.** `tldw_chatbook/Research_Workspace/__init__.py`,
`tldw_chatbook/tldw_api/notes_workspace_limits.py` (new),
`tldw_chatbook/tldw_api/notes_workspace_schemas.py`,
`tldw_chatbook/Research_Workspace/server_adapter.py`, `tldw_chatbook/app.py`,
`Tests/Packaging/test_research_workspace_import_closure.py` (new, 5 tests),
`Tests/Performance/test_ui_ready_module_census.py`,
`Tests/App/test_construct_runtime_imports.py`.

**Known residual.** The default-on background screen pre-importer still warms every screen
module (this family included) on a daemon thread seconds after first paint -- deliberate,
off the critical path, finding 22214's territory, pinned OFF in the probes exactly as the
census guard documents.
