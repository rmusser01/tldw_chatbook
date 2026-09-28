---
id: TASK-32943
title: F6 moves between panes on Settings
status: Done
created_date: 2026-09-24 12:00
assignee:
- '@claude'
labels:
- settings
- keyboard
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The footer advertises "F6 next pane" on Settings, but pressing it only toasted "No workbench pane focus target is available": `SettingsScreen` had no `action_focus_next_workbench_pane`, so the app-global F6 fell through to its notice. Library, Console, Personas and Workflows all cycle their panes; Settings should too, and the user guide claimed F6 "does nothing" there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 F6 on Settings cycles focus category rail -> detail pane -> Scope Inspector -> rail
- [x] #2 Shift+F6 cycles the same panes in reverse
- [x] #3 Landing on the rail focuses the active category button; a pane with nothing focusable is skipped
- [x] #4 User guide describes the new keys
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add `action_focus_next/previous_workbench_pane` on SettingsScreen (app F6 delegates to the former)
2. Resolve each pane's target from the screen focus chain (the shared `focus_relative_workbench_pane` helper needs fixed child ids; Settings' detail content changes per category)
3. Add a screen shift+F6 binding (Personas/Library precedent)
4. Pilot test; update Docs/User_Guide/settings.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
`SettingsScreen._focus_relative_settings_pane` walks `self.focus_chain` for the first focusable member of `#settings-category-pane`, `#settings-detail-pane` and `#settings-impact-pane` (rail prefers `#settings-category-<active>`), skips empty panes, and cycles by the pane containing the current focus. Did not reuse `Widgets/workbench_focus.focus_relative_workbench_pane`: it resolves targets by preferred ids, and the detail pane's first control differs per category. Shift+F6 is a screen `Binding(priority=True)` like Personas/Library.

Test: `test_settings_f6_cycles_rail_detail_inspector` (hub file, `@private_profile_test`). The DestinationHarness app has no app-level F6 binding, so the test calls the delegate action for F6 and presses the real shift+F6.

Files: `tldw_chatbook/UI/Screens/settings_screen.py`, `Tests/UI/test_settings_configuration_hub.py`, `Docs/User_Guide/settings.md`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
