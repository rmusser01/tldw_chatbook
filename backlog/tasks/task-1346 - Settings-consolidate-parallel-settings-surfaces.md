---
id: TASK-1346
title: Settings consolidate parallel settings surfaces
status: Done
assignee: []
created_date: '2026-08-04 23:47'
updated_date: '2026-08-05 15:16'
labels:
  - settings
  - cleanup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three settings surfaces coexist with different labels and save behavior: the Settings screen, the 7335-line legacy Tools_Settings_Window.py, and enhanced_settings_sidebar.py inside the legacy Chat window; minimal_settings_sidebar.py (256 lines) has no importers and is dead code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 minimal_settings_sidebar.py is removed or justified,Legacy Tools and Settings window is retired or clearly scoped with a migration note,A single canonical settings surface is documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify minimal_settings_sidebar.py has no importers (confirmed: zero .py references, incl. Tests and dynamic strings) and delete it
2. Verify Tools_Settings_Window.py reachability (confirmed: tools_settings route resolves to MCPScreen in UI/Navigation/screen_registry.py; nothing composes id=tools_settings-window; wrapper ToolsSettingsScreen is registered but unrouted). Deletion is NOT trivially safe (app.py module-level import + IngestUiStyleChanged handler + Tests/UI/test_tools_settings_window.py exercises it directly) -> scope it with a deprecation/migration banner instead
3. Add deprecation note to tools_settings_screen.py wrapper docstring
4. Document canonical settings surface: settings_screen.py module docstring + AGENTS.md UI section
ADR required: no
ADR path: N/A
Reason: dead-code removal plus documentation only; no runtime route, storage, or interface boundary changes (the tools_settings->MCPScreen route already exists and is untouched)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Deleted tldw_chatbook/Widgets/minimal_settings_sidebar.py (verified zero importers across tldw_chatbook/ and Tests/, no dynamic-import strings). Scoped the legacy Tools_Settings_Window.py instead of deleting: it is unreachable via navigation (the tools_settings route resolves to MCPScreen in UI/Navigation/screen_registry.py, and nothing composes id=tools_settings-window), but deletion was not trivially safe (app.py module-level import + IngestUiStyleChanged handler, Tests/UI/test_tools_settings_window.py exercises it directly). Added DEPRECATED/migration banners to Tools_Settings_Window.py and the unrouted tools_settings_screen.py wrapper pointing to the canonical surface. Documented the canonical settings surface (UI/Screens/settings_screen.py, F9) in its module docstring and AGENTS.md UI section. ADR: not required (dead-code removal + docs; no route/boundary change). Tests: pytest Tests/UI/test_destination_shells.py Tests/UI/test_tools_settings_window.py -q -> 115 passed, 17 skipped, 1 failed (test_destination_action_buttons_explain_their_outcome[library] — known pre-existing failure, unrelated). Import smoke of app.py + touched modules passes.
<!-- SECTION:NOTES:END -->
