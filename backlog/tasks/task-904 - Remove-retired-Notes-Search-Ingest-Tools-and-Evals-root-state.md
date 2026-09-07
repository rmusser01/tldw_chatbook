---
id: TASK-904
title: Remove retired Notes Search Ingest Tools and Evals root state
status: Done
assignee: []
created_date: '2026-07-26 23:50'
updated_date: '2026-07-27 22:26'
labels:
  - architecture
  - state
  - cleanup
dependencies:
  - TASK-647
references:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - backlog/decisions/033-application-session-state-ownership.md
  - >-
    Docs/superpowers/specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Delete unreachable or no-op root reactives and companion defaults and timers for production destinations that already own their state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Notes field, sort, preview, and autosave state; Search active-subtab state; Ingest active-view state; Tools active-view state; and Evals sidebar state are removed with every writer, watcher, initializer, timer, and dynamic reference.
- [x] #2 Library remains the production owner for Notes and the retired Ingest route's Import media canvas, while Search, MCP, and Evals remain the only owners of their respective view state.
- [x] #3 No compatibility root properties or mirrored state are introduced.
- [x] #4 The normal production TldwCli can navigate to and exercise Search and Evals plus the Notes, Ingest, and Tools legacy aliases at their current Library/MCP owners without removed-name access.
- [x] #5 Focused ownership, static, formatting, compile, and authorized integration checks pass.
- [x] #6 The stream specification and remaining TASK-905/TASK-906 tasks and plans use their actual task IDs, dependencies, filenames, backlog CLI targets, and current production routes rather than unrelated TASK-654/TASK-655 records or the retired MediaIngestScreen/api_calls pipeline.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md
Reason: Existing ADRs and the current route registry assign Notes and the retired Ingest route to Library, Search to SearchScreen, legacy Tools to MCP, and Evals to EvalsScreen.

1. Add exact removed-name guards.
2. Navigate every production destination and current legacy alias.
3. Delete remaining root fields, timers, watchers, and initializers while keeping already-retired paths absent.
4. Reconcile remaining stream task/plan IDs and latest-dev production routes, then verify destination-owned state remains.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Rebased the completed change onto `origin/dev` at
  `4148e148cfb8541a2a819e5e210b8885974fe7ec` and reran all final evidence
  against that exact tree.
- Removed the retired Notes, Search, Ingest, and Tools root fields, defaults,
  watchers, timer/cleanup paths, and Notes root switch dispatch. Kept the
  already-absent Evals field and `Event_Handlers/tab_initializers` package
  pinned absent. Also removed the directly orphaned lazy placeholder,
  rebuilt-Ingest aliases, all six old Search navigation constants, and their
  unused widget imports.
- Added root-aware AST sentinels for direct, chained, dynamic, mapping,
  selector, and definition access. Mutation tests proved the guards detect
  `TldwCli` ownership without rejecting same-named destination or nested-owner
  fields/methods.
- Added a normal mounted `TldwCli` production test. Two independent app
  lifecycles navigate Library/Notes/Search/Ingest/MCP/Tools/Evals, exercise
  destination-owned state and restoration, and rely on Textual's normal
  `run_test()` teardown. No test/simplified application is used.
- Reconciled the stream specification and plan filenames from obsolete
  TASK-653/654/655 records to TASK-904/905/906. Latest-dev inspection proved
  the old `api_calls` producer, MediaIngestScreen, widgets, and
  `tldw_api_events.py` are already absent, so TASK-905 now deletes the orphan
  context/routing/handlers rather than rebuilding a result-envelope pipeline.
  Independent specification re-review reported `SPEC COMPLIANT`.
- TDD evidence: the initial exact ownership command failed with `2 failed, 43
  passed`; mutation tests then failed on chained roots, dynamic storage,
  selectors, destination false positives, nested-owner false positives, and
  the exact orphan companions before their respective fixes.
- Final verification:
  - `pytest Tests/ProductionApp/test_retired_destination_root_state.py
    Tests/test_application_state_ownership.py -q` — `50 passed, 2 warnings in
    234.04s`.
  - `pytest Tests/ProductionApp Tests/test_application_state_ownership.py -q`
    — `87 passed, 4 warnings in 408.01s`.
  - `python -m compileall` over the changed app/test and affected destination
    imports — passed.
  - `python -m ruff check` over the changed app/tests and affected destination
    screens — passed.
  - `python -m ruff format --check` over both changed tests — passed. The two
    unchanged upstream format exceptions (`library_screen.py` and
    `evals_screen.py`) remain byte-identical to `origin/dev`; no unrelated
    formatting was applied.
  - Exact retired-symbol and initializer import/package scans — no matches;
    `git diff --check` — passed.
- The remaining warnings are existing Requests dependency-version and
  `datetime.utcnow()` deprecation warnings outside this task's changed code.
- ADR required: yes. Reused
  `backlog/decisions/033-application-session-state-ownership.md` and
  `backlog/decisions/011-chatbook-workbench-ui-system.md`; no new decision was
  introduced.
<!-- SECTION:NOTES:END -->
