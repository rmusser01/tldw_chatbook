---
id: TASK-2082
title: 'Roleplay: design the first paint (auto-select, honest inspector) (F-031)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-04 08:16'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Nothing selected on mount: center void, 5 disabled buttons, disabled checkbox, and a false 'Validation: OK'. Evidence: personas_inspector_pane.py:138. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 First library row is auto-selected on mount when the library is non-empty,With no selection the action stack and Validation line are hidden and a single guidance line shows,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests first: inspector pane hides action stack/Conversations+Readiness headers/Validation pre-selection and shows one guidance line; screen auto-selects first library row on mount (fresh mounts only - restore round-trips and empty libraries exempt). 2. PersonasInspectorPane: id the Conversations/Readiness headers, gate their display plus the action stack and Validation line on _has_selection inside _apply_action_state. 3. PersonasScreen._load_after_mount: after _apply_pending_restore, auto-select first row via the existing _select_character path (no focus moves - focus-steal guards untouched); restore_state marks saved-state mounts so they skip auto-select. 4. Update tests pinning the old no-selection first paint. ADR required: no - UI composition/copy change on existing selection flow; no schema, boundary, or contract change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented (a) mount-time auto-select of the first library row via the existing _select_character path (no focus moves; restore_state marks saved-state mounts via _restored_from_saved_state so Console round-trips keep their own selection semantics; empty libraries and non-characters modes skip), plus a _sync_title_and_console_actions after auto-select so header/footer reflect the selection on first paint; (b) inspector hides the action stack, Conversations+Readiness headers, conversations list, and Validation line when kind is None and shows one guidance line ('Pick a character or persona to start chatting.') via _apply_action_state; (c) Validation display is selection-gated so no false 'Validation: OK' pre-selection. Files: tldw_chatbook/UI/Screens/personas_screen.py, tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py; tests in Tests/UI/test_personas_{inspector_pane,workbench,workbench_state}.py + test_product_maturity_phase1_empty_setup_states.py. Unsaved-edit guards verified intact (guarded paths untouched; auto-select bypasses the guard only because first paint has no edits; test_first_paint_auto_select_keeps_unsaved_guards_quiet pins clean follow-up selection). Verified: 326 workbench/state/inspector tests pass, plus dictionary/lore/preview/library-pane/footer-hints suites (202) and ProductionApp/navigation/phase6 (86); ruff clean. ADR: not required (UI composition on existing selection flow).
<!-- SECTION:NOTES:END -->
