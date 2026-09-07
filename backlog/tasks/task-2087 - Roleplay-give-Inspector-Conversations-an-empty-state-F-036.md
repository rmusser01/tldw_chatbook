---
id: TASK-2087
title: 'Roleplay: give Inspector Conversations an empty state (F-036)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-04 10:12'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
show_conversations(()) is called without empty_copy, leaving a dangling 'Conversations' header pre-selection. Evidence: personas_inspector_pane.py:198-209. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Conversations section shows empty copy or is hidden when empty,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing pane tests: character selection with zero conversations renders 'No saved conversations.' placeholder; persona/dictionary/lore selections hide the Conversations section entirely (kind-aware, matching the task-443 inspector idiom). 2. PersonasInspectorPane._apply_action_state: conversations header+list display = selected AND kind==character. 3. personas_screen._select_character server branch passes empty_copy='No saved conversations.' (the local path already does via the controller). 4. Run pane + conversations suites + ruff. ADR required: no - inspector section visibility/copy; no behavior-contract change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The Conversations section now follows the inspector's existing kind idiom (task-443): _apply_action_state shows the header+list only for character selections (personas/dictionaries/lore have no conversation linkage - hiding matches how the console/export actions are handled for those kinds), and characters with zero saved conversations render the 'No saved conversations.' placeholder in every path (local already did via the controller; the server select branch now passes the same empty_copy). Pre-selection hiding from task-2082 is unchanged. Files: tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py, tldw_chatbook/UI/Screens/personas_screen.py; tests in Tests/UI/test_personas_inspector_pane.py (kind-gate + empty-copy rendering). Verified: pane 26 passed; gate 324 passed (pane + full workbench incl. server isolation and conversations panel classes); ruff clean. ADR: not required (section visibility/copy).
<!-- SECTION:NOTES:END -->
