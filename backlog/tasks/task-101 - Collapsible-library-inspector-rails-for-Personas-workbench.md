---
id: TASK-101
title: Collapsible library/inspector rails for Personas workbench
status: Done
assignee: []
created_date: '2026-06-11 18:26'
updated_date: '2026-06-26 17:58'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Adopt ConsoleRailHandle (as Console and Notes do) for the Personas workbench library and inspector panes so they collapse like Console rails. Deferred from the UX parity pass as a structural compose change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library and inspector collapse/expand via rail handles,Keyboard reachable,Tests cover collapsed state
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify latest origin/dev does not already include Personas rail collapse behavior.
2. Reuse ConsoleRailHandle for collapsed Library and Inspector affordances.
3. Add open-state collapse buttons to the Personas Library and Inspector pane headers using the existing Console/Notes rail header vocabulary.
4. Keep rail collapsed state screen-local and sync pane/handle visibility without changing storage or shared Personas data contracts.
5. Add mounted UI tests covering collapsed state, reopen behavior, and keyboard reachability.
6. Run the focused Personas workbench suite and document results.

ADR required: no
ADR path: N/A
Reason: This is a contained Textual UI composition parity change that reuses the existing ConsoleRailHandle pattern and does not alter storage, schema, sync policy, service boundaries, security, runtime contracts, or dependency choices.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added shared `ConsoleRailHandle` affordances for the Personas Library and Inspector panes, with screen-local collapsed state and explicit visibility sync for panes and handles. Added matching open-state collapse buttons to the Library and Inspector pane headers using the existing Console/Notes rail header classes. Covered the behavior with mounted UI tests for collapse/reopen, collapsed-state visibility, keyboard reachability through Shift+F6, and Personas-specific collapsed-handle tooltips.

PR review follow-up rebased the branch onto the latest `origin/dev`, switched rail visibility updates to Textual's `Widget.display` property, added Google-style docstrings to modified public compose methods, moved collapse-button fixed width into TCSS, and replaced repeated rail handle width literals with named constants.

Verification:
- `.venv/bin/python -m pytest Tests/UI/test_personas_workbench.py` -> 138 passed
- `git diff --check` -> passed
- ADR required: no; no storage/schema/service/runtime/security/dependency boundary changed.
<!-- SECTION:NOTES:END -->
