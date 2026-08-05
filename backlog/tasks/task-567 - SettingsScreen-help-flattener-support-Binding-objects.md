---
id: TASK-567
title: 'SettingsScreen help flattener: support Binding objects'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 07:57'
updated_date: '2026-07-25 16:24'
labels:
  - settings
  - tech-debt
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
SettingsScreen.action_show_workbench_help (541 AC6 fix) flattens only tuple/list BINDINGS entries; a future Binding(...) entry would silently vanish from the F1 help with no test failing. Forward-compat only — all current entries are tuples.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Binding instances are rendered in the screen help output
- [x] #2 Regression test covers a mixed tuple+Binding BINDINGS list
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a regression test: monkeypatch a mixed tuple+Binding SettingsScreen.BINDINGS list and assert action_show_workbench_help renders a row for each.
2. Update the flattener in action_show_workbench_help to extract (key, action, description) from a Binding instance the same way it does for tuple/list entries.
3. Run the RAG profile region test file green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added SettingsScreen._binding_entry_key_action_description, a static helper extracting (key, action, description) from a BINDINGS entry that is either the tuple/list shape or a Binding(...) instance -- Textual's two valid BINDINGS entry shapes. action_show_workbench_help's flattener now runs every self.BINDINGS entry through this helper (skipping unrecognized shapes) before the existing RAG-accelerator filter, instead of only handling isinstance(entry, (tuple, list)).

Regression test in Tests/UI/test_settings_rag_profile_region.py: test_action_show_workbench_help_flattens_binding_instances_too monkeypatches SettingsScreen.BINDINGS with a mixed tuple + Binding(...) list and asserts both descriptions ("Undo edit", "Redo edit") appear in the rendered WorkbenchHelpPanel state -- verified RED (only the tuple entry survived pre-fix) then GREEN.

Full Tests/UI/test_settings_rag_profile_region.py: 116 passed. Also re-ran Tests/UI/test_workbench_focus_help.py and Tests/UI/test_settings_rag_profile_adapter.py (74 passed) since both touch the shared help-flattening/BINDINGS machinery -- no regressions.
<!-- SECTION:NOTES:END -->
