---
id: TASK-112
title: Debounce Personas library search input
status: Done
assignee: []
created_date: '2026-06-11 03:23'
labels:
  - personas
  - search
  - performance
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Search re-renders per keystroke; add ~200ms debounce in the search pipeline to cut render churn and FTS query volume.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typing quickly triggers at most one render per debounce window,Final query always renders
<!-- AC:END -->

## Implementation Plan

1. Trace the current Personas library search pipeline from `Input.Changed` through screen-level row rendering.
2. Add a mounted regression that changes the search input rapidly and asserts intermediate changes do not trigger multiple renders before the debounce window completes.
3. Add a companion assertion that the final query always renders after the debounce window.
4. Implement a small debounced search worker/timer path that preserves mode-specific rendering and cancels stale work on mode changes.
5. Run focused Personas search tests plus diff hygiene, then update acceptance criteria and implementation notes.

ADR required: no
ADR path: N/A
Reason: bounded UI debounce/performance fix; no storage/schema, sync policy, service boundary, or long-lived architecture decision.

## Implementation Notes

- Added a 200ms screen-owned debounce timer for Personas library search changes.
- Search renders now run through a single exclusive `personas-library-search` worker and ignore stale mode/query snapshots.
- Pending search timers are cancelled on mode changes and unmount to prevent late renders into the wrong mode.
- Added a mounted regression proving rapid search changes do not render before the debounce window and the final query renders once.
- Updated existing Personas search tests to wait through the debounce window deterministically.
- Review follow-up: row render helpers now re-check the captured query/mode after awaited FTS work and before updating the library pane, so stale search results cannot render into a newer query or mode.
- Added a mounted regression for stale FTS results and typed/documented the debounce regression called out in review.
- Verification: `.venv/bin/python -m pytest -q Tests/UI/test_personas_workbench.py --tb=short` passed with 140 tests; the four UI tests that failed on the old PR head passed locally after the rebase; `git diff --check` passed.
- ADR check completed: no ADR required for this bounded UI interaction/performance fix.
