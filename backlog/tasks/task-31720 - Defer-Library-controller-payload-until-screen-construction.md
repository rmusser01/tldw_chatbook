---
id: TASK-31720
title: Defer Library controller payload until screen construction
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 17:48'
updated_date: '2026-09-05 17:56'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep unvisited Library controller implementations outside the startup route-discovery import closure while preserving first-use behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library route preimport stays below existing payload budgets without raising ceilings.
- [x] #2 Deferred implementations resolve to their original classes on first use and existing event and callback identities remain intact.
- [x] #3 Focused first-use, controller behavior and architecture checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the current fresh-process route-import closure and census direct uses of the Notes sync controller/runtime imports.
2. Add an isolated regression proving Library route discovery does not import Notes synchronization execution and that real screen construction resolves original controller/runtime implementations.
3. Move the existing Notes sync import block into LibraryScreen construction, preserving call sites, named ports, event identities and screen size ceilings. Measure whole-pass and Library marginal changes; expand only if measurements show the requested savings are not achieved.
4. Verify focused Notes sync/Library construction tests, packaging closure and architecture guards; leave coordinated global snapshot refresh to root.
ADR required: no
ADR path: backlog/decisions/097-boot-budget-ratchets.md
Reason: Direct application of the existing first-use import deferral requirement; no new runtime boundary, provider contract or dependency.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved only the existing Notes sync controller import block from module scope into LibraryScreen construction. The same LibraryNotesSyncController and InertLastingSyncRuntime classes, constructor ports and app-owned runtime lifecycle remain unchanged; no proxy or new lifecycle mechanism was introduced. Source size remains 41,324 lines / 1,301 methods.
Fresh Library marginal preimport payload fell from 188 modules / 146,804 LOC to 178 modules / 130,426 LOC: 10 modules / 16,378 LOC deferred. The combined current pass measures 489 modules / 371,218 LOC, including concurrent Settings/Scheduling savings; this aggregate delta is not attributed solely to Library. Existing full preimport budget test passes (1 passed, 5.98s). Root owns final global snapshot refresh and ADR tightening.
The isolated regression failed before the change with all four forbidden execution modules resident, then passed after the change and constructed a real LibraryScreen with exact original controller/runtime identity. Focused packaging, controller, Library flow, production lifecycle and screen architecture selection: 148 passed, 3 deselected, 46.97s. New test Ruff/format and diffcheck pass. Parent reviewed the exact import relocation with no actionable finding.
ADR required: no new ADR; applies backlog/decisions/097-boot-budget-ratchets.md. Library decomposition recipe section21 documents the measured incident and scope.
<!-- SECTION:NOTES:END -->
