---
id: TASK-1377
title: Remove dead hidden per-category status rows
status: Done
assignee: []
created_date: '2026-08-05 23:38'
updated_date: '2026-08-05 23:53'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique minor: per-category 'Status:' rows render then are permanently hidden (.settings-category-status-hidden) — dead UI shipped to every category. Remove the rows and their machinery (or surface them if they were intended to be visible — check git history first).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No permanently-hidden status rows rendered in any settings category
- [x] #2 CSS class .settings-category-status-hidden removed with its widgets
- [x] #3 No test references the removed machinery (see Implementation Notes: one out-of-scope reference handed off)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Git history: rows were born hidden in 2f2da334c (#366) - genuinely dead
2. Remove the status Static from _render_category_buttons and its update block in _update_draft_status_widgets
3. Remove .settings-category-status-hidden (and now-unused .settings-dirty-category) from components/_agentic_terminal.tcss, regenerate bundle
4. Sweep Tests for references; keep _category_status (still used by category search ranking)
ADR required: no
ADR path: N/A
Reason: dead-UI removal, no architectural decision
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Git history check: `git log -S settings-category-status-hidden` shows the rows were introduced already hidden in 2f2da334c ("Polish Settings configuration hub (#366)", May 2026) and never made visible — genuinely dead UI.
- Removed the per-category `Status:` Static from `_render_category_buttons` and its update block in `_update_draft_status_widgets` (`tldw_chatbook/UI/Screens/settings_screen.py`). `_category_status()` is kept: it still feeds `_category_search_rank` (secondary search haystack).
- Removed the `.settings-category-status-hidden` rule from `tldw_chatbook/css/components/_agentic_terminal.tcss` and regenerated `tldw_cli_modular.tcss`. The `.settings-dirty-category` rule was deliberately KEPT — it still styles the visible `#settings-category-state-banner` (`_render_category_state_banner`/`_update_category_state_banner`), and dirty state remains visible via the banner and the `*` in the category button label.
- Test added: `test_settings_category_rail_renders_no_hidden_status_rows` asserts no `.settings-category-status-hidden` nodes and no `settings-category-*-status` Statics in the rail.
- Hand-off: `Tests/UI/test_destination_visual_parity_correction.py::test_settings_dirty_category_status_has_visual_marker_class` (outside this task's editable file set) still asserts on the removed `#settings-category-console-behavior-status` widget and now fails; it should be repointed at the visible `#settings-category-state-banner` (same `settings-dirty-category` class) by whoever owns that file.
- Files: `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/css/components/_agentic_terminal.tcss`, `tldw_chatbook/css/tldw_cli_modular.tcss` (generated), `Tests/UI/test_settings_configuration_hub.py`.
<!-- SECTION:NOTES:END -->
