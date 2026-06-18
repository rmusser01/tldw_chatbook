---
id: TASK-116
title: Unify Console-action gating between Personas screen and inspector pane
status: Done
assignee: []
created_date: '2026-06-11 04:54'
updated_date: '2026-06-18 15:00'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
personas_screen._console_action_allowed and PersonasInspectorPane._apply_action_state derive enablement independently; when prompts/dictionaries/lore selection lands they will diverge. Push a single set_console_actions_enabled(bool) from the screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One source of truth for attach/start-chat enablement
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a bounded UI state-ownership correction inside the existing Personas/Console handoff contract; it does not change storage, provider/runtime boundaries, service contracts, or long-lived architecture.

1. Add a failing mounted inspector regression proving Console attach/start enablement is not derived by the inspector from selection alone.
2. Add a single inspector setter for screen-owned Console action availability while preserving inspector-local export/delete selection state.
3. Push PersonasScreen._console_action_allowed() into the inspector during footer/selection/editing state refreshes.
4. Add mounted PersonasScreen coverage proving the screen gate controls visible Attach/Start state after selection.
5. Run focused Personas UI verification and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a single screen-owned Console action gate for Personas attach/start actions. PersonasInspectorPane now exposes set_console_actions_enabled(), keeps export/delete availability local to selected/unsaved state, and no longer enables Attach/Start from show_selection() alone. PersonasScreen now pushes _console_action_allowed() plus user-facing block reasons into the inspector during footer/selection/editing state synchronization.

Added regressions covering the new inspector ownership boundary and a mounted PersonasScreen guard that forces the screen gate false after selection and verifies the visible Attach/Start buttons plus readiness copy follow the screen-owned state.

Verification: python -m pytest -q Tests/UI/test_personas_inspector_pane.py Tests/UI/test_personas_workbench.py Tests/UI/test_personas_workbench_foundation.py --tb=short -> 142 passed. git diff --check -> passed.

PR review follow-up: pushed the screen-owned Console gate immediately after selection/save state mutations and before subsequent async loading/rendering work, preventing a transient inspector state where a valid selected item rendered with blocked Attach/Start actions. Added mounted timing regressions for character selection, character save reload, and profile save row rendering. Added Google-style docstring sections requested by review.

Review verification: python -m pytest -q Tests/UI/test_personas_workbench.py::TestConsoleActions::test_selection_pushes_console_gate_before_async_followup Tests/UI/test_personas_workbench.py::TestConsoleActions::test_character_save_pushes_console_gate_before_reload Tests/UI/test_personas_workbench.py::TestConsoleActions::test_profile_save_pushes_console_gate_before_row_render --tb=short -> 3 passed. python -m pytest -q Tests/UI/test_personas_inspector_pane.py Tests/UI/test_personas_workbench.py Tests/UI/test_personas_workbench_foundation.py --tb=short -> 145 passed.

ADR required: no; bounded UI enablement ownership correction only.
<!-- SECTION:NOTES:END -->
