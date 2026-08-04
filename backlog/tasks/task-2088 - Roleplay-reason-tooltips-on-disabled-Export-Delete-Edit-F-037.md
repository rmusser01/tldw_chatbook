---
id: TASK-2088
title: 'Roleplay: reason tooltips on disabled Export/Delete/Edit (F-037)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-04 10:26'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Export JSON/PNG, Delete, and card Edit are disabled with no reason in the no-selection state; Attach/Start Chat show the right pattern. Evidence: personas_inspector_pane.py:406-414. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All disabled inspector actions carry a reason tooltip,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests first: pane default state asserts every disabled action carries a reason tooltip; server-browsing state asserts Edit/Export/Delete tooltips; card Edit id-less state tooltip. 2. PersonasInspectorPane._apply_action_state: no-selection reason tooltips for export/delete, intent tooltip for console actions (guidance replaces stale 'Console action blocked: select an item'). 3. personas_screen._sync_local_character_actions + server select branch: set a read-only reason tooltip when force-disabling Edit/Export/Delete in server mode (and clear on restore). 4. Card widget: Edit tooltip for the no-saved-id case. 5. Check the skipped tooltip audit at Tests/UI/test_destination_shells.py:2203 for re-enable scope. ADR required: no - tooltip/copy additions on existing gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Every disabled inspector action now carries a reason tooltip: no-selection -> intent guidance (console pair), 'Select an item to export.' (exports), 'Select an item to delete.' (delete); unsaved edits keep the save-first tooltip; the screen-blocked custom-reason tooltip now matches the readiness line's intent copy ('Chat blocked: <reason>'). Server browsing force-disables card Edit + inspector export/delete with 'Server characters are read-only here.' (set in _sync_local_character_actions and the server select branch; cleared on restore). Card Edit also explains its no-selection and no-saved-record states. Deferral: the test_destination_shells all-buttons tooltip audit (line ~2203) stays skipped - it audits tooltips on EVERY button incl. always-enabled editor Save/Cancel and preview controls, beyond F-037's disabled-reason scope. Files: personas_inspector_pane.py, personas_character_card_widget.py, personas_screen.py; tests in test_personas_{inspector_pane,character_widgets,workbench}.py. Verified: 60 pane+card tests, server-isolation tooltip test, gate 464 passed/1 skipped (pane, card widgets, full workbench, UAT, destination shells); ruff clean. ADR: not required (tooltips on existing gates).
<!-- SECTION:NOTES:END -->
