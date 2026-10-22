---
id: TASK-35647
title: Settings Overview and the Tools & MCP state badge still describe the retired MCP screen
status: Done
assignee:
- '@claude'
created_date: 2026-10-20 03:07
updated_date: 2026-10-22 20:24
dependencies: []
labels:
- follow-up
- settings
- copy
priority: low
references:
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/UI/Screens/settings_tools_hub.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of Phase 9's rebase onto dev 88fe3108e5 (2026-10-20). task-35150 (#3140) retired the MCP destination into Settings ▸ Tools & MCP, but two pieces of Settings copy still describe the old layout. The Overview card's boundary rows (`SETTINGS_OVERVIEW_BOUNDARY_ROWS` in settings_screen.py) still say "MCP: tool servers are managed on the MCP screen", a screen that no longer exists. And `_persistence_badge` has no Tools & MCP branch, so the category's State banner falls through to "Read-only here", while the category's own summary (`TOOLS_HUB_CATEGORY_SUMMARY`) calls its save model "Immediate actions". A user reading the Overview is sent to a missing screen, and the Tools & MCP banner contradicts the category list.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Settings Overview names where tool servers are managed now (Settings ▸ Tools & MCP) and names no retired screen
- [x] #2 The Tools & MCP State banner and the category's summary name the same save model
- [x] #3 A test fails if either Overview row or the Tools & MCP badge names a destination that is not registered
<!-- AC:END -->

## Implementation Plan

1. Overview boundary row: name Settings ▸ Tools & MCP as where tool servers are managed. The Overview's mode strip ("Runtime controls stay in MCP and ACP") also names the retired MCP destination, so it names Tools & MCP too (AC#1 covers the whole Overview).
2. `_persistence_badge`: a Tools & MCP branch that returns `TOOLS_HUB_CATEGORY_SUMMARY.status`, so the banner and the category list read one definition. settings_screen.py sits at its module-size row (35,668); the two added lines are paid for by folding two identical "Applies immediately" branches.
3. A test in Tests/UI/test_settings_tools_hub.py (a whole-file UI-lane census entry) that scans every Overview boundary row and the Tools & MCP badge for "Settings ▸ <X>" (X must be a Settings category title) and "the <X> screen" (X must resolve through `screen_registry` and must not be a `_SCREEN_ALIASES` key, which is the list of retired screens; "mcp" resolves only as an alias). Includes a negative control on the old row text, shown red before the copy fix.
4. Update the pinned copy tests (test_settings_configuration_hub.py, test_destination_shells.py) and Docs/User_Guide/settings.md.

## Implementation Notes

Settings copy now names Settings ▸ Tools & MCP where it used to name the retired MCP screen. The Tools & MCP State badge reads the category list's own save model.

- **Overview** (`settings_screen.py`): the boundary row reads "MCP: tool servers are managed in Settings ▸ Tools & MCP". The Overview mode strip also pointed at the retired destination ("Runtime controls stay in MCP and ACP"), so it now reads "Runtime controls are in Tools & MCP and ACP". AC#1 is about the whole Overview, so the strip is in scope.
- **Badge**: `_persistence_badge` has a Tools & MCP branch that returns `TOOLS_HUB_CATEGORY_SUMMARY.status`. The State banner and the category list both read that one record, so the banner reads "State: Immediate actions | …". settings_screen.py is at its module-size row (35,668 lines). The two lines the branch adds are paid for by folding the identical Schedules and Dreams "Applies immediately" branches into one, so the file stays at 35,668.
- **Test (AC#3)**: `Tests/UI/test_settings_tools_hub.py::test_overview_rows_and_tools_badge_name_only_registered_destinations`. This file is a whole-file UI-lane census entry. The test scans every `SETTINGS_OVERVIEW_BOUNDARY_ROWS` row (label and value) and the Tools & MCP badge for two forms:
  - "Settings ▸ <X>": X must be the title of a category in `_category_summaries()`.
  - "the <X> screen": X must resolve through `screen_registry.resolve_screen_route` and must not be a `_SCREEN_ALIASES` key. An alias is a retired screen's redirect. "mcp" resolves only as an alias (to settings), so a route-only check would have passed the old row.

  The test also pins AC#2: the badge equals the summary's status, the summary in the category list is that same record, and the banner starts with it. Built-in negative controls, part of the test itself: the old row text, and a made-up "Settings ▸ MCP Servers".
- **Covering runs** (all through the scratch wrapper, `-n 4`):
  - test_settings_tools_hub.py, test_settings_footer_hints.py and test_settings_state_line_unsaved_count.py: 42 passed.
  - The private-profile Overview and mode-line pins in test_settings_configuration_hub.py pass.
  - The rest of that file's `-k "overview or banner or mode_line or help or persistence or ownership"` selection ran 29 passed and 19 failed. All 19 are plain (non-private-profile) tests, and every one failed at setup with `RecoveryRequired: raw_source_selection_changed`, the known local signature. Eight of them cover the banner or ownership records; under a temporary `@private_profile_test` (reverted), 7 of those 8 passed. The eighth, `test_settings_domain_categories_are_grouped_and_have_ownership_records`, fails the same way at dev 28beeb9ed5 (`assert not record.writes_allowed`), so the failure predates this change.
  - `test_settings_theme_picker_screen.py::test_try_then_clean_category_switch_restores_the_pre_editor_theme` failed once in a `-n 4` run, with Textual Select `NoMatches '#label'`. Run alone it passed 3 of 3 times.
  - Module size ratchet for settings_screen.py: passed.
  - `./scripts/preflight.sh`: passed.
- **Red/green**:
  - Against the old row, the test failed with `assert ['the MCP screen'] == []` on "MCP: tool servers are managed on the MCP screen".
  - With the row fixed and before the badge branch, it failed with `assert 'Read-only here' == 'Immediate actions'`.
  - With both fixes it passes.
- **Pinned copy tests updated on purpose**:
  - `test_settings_configuration_hub.py::test_settings_overview_renders_ownership_contract_boundaries`: new row, plus no "MCP screen" anywhere on the Overview.
  - `test_settings_configuration_hub.py::test_mode_line_disclaimer_only_on_overview`.
  - `test_destination_shells.py::test_settings_destination_uses_three_column_workbench_contract`. This test is plain, not private-profile. Locally it fails at dev 28beeb9ed5 with `RecoveryRequired: raw_source_selection_changed` (the known local signature). The new pin passed once under a temporary `@private_profile_test`, which was reverted.
- **Docs**: Docs/User_Guide/settings.md has the new mode-strip copy, and its badge table has an "Immediate actions" row for Tools & MCP.
- **Files**: tldw_chatbook/UI/Screens/settings_screen.py, Tests/UI/test_settings_tools_hub.py, Tests/UI/test_settings_configuration_hub.py, Tests/UI/test_destination_shells.py, Docs/User_Guide/settings.md.
