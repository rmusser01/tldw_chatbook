---
id: TASK-2829
title: Settings user-language copy pass
status: Done
assignee: []
created_date: '2026-08-04 23:47'
updated_date: '2026-08-05 15:18'
labels:
  - settings
  - ux
  - copy
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Engineering vocabulary leaks into user-facing copy: 'Automatic refresh (ADR-020)' (:6268), 'Domain ownership contract', 'persisted preference contract', dev-backlog 'Follow-up:' text (:452-577), a title promising 'accounts' that don't exist (:7721), palette entry 'Open Settings Tab' described as 'Navigate to Tools & Settings tab' (app.py:1070-1072), and Theme Delete's 'Cannot delete built-in or custom themes' which makes Delete a near-always dead end (settings_theme_editor.py:550-554).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No ADR numbers, backlog follow-up text, or contract vocabulary in user-facing labels
- [x] #2 Palette entries accurately name their destination
- [x] #3 Theme Delete guard distinguishes built-in from deletable custom themes with accurate copy
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read current user-facing copy sites (settings_screen.py, app.py palette, settings_theme_editor.py delete guard)\n2. Reword labels/banners into user language (drop ADR refs, contract vocabulary, backlog follow-up notes); fix palette destination names; make Theme Delete guard distinguish built-in vs custom with accurate copy\n3. Update tests pinning old copy\n4. Run target suites; green except known pre-existing failures\nADR required: no\nADR path: N/A\nReason: copy-only change, no architectural decision
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Copy-only pass over Settings user-facing strings; no behavior changes except the Theme Delete guard.

Approach: swept settings_screen.py, app.py, and settings_theme_editor.py for the listed engineering vocabulary (ADR references, "contract"/"mutation" language, dev-backlog "Follow-up:" notes, stale destination names) and reworded each site in user language, keeping technical precision where it is intentional (inspector "Saved as:" config keys, internal identifiers, code comments).

Key decisions:
- Dropped "(ADR-020)" from the "Automatic refresh" section header and the two model-catalog inspector "Purpose" strings.
- Reworded the 8 read-only domain `follow_up` values from "Follow-up: add X after Y exposes a persisted ... contract" to "Editing ... from Settings is planned for a future release."; the contract DATA (owner, sources, boundary rows) is unchanged per task context.
- Renamed Domain Ownership detail labels: "Domain ownership contract" -> "Domain ownership", "Owner destination" -> "Managed in", "Source of truth" -> "Where the data lives", "Follow-up" -> "Planned"; de-contracted "Settings mode"/"Writes allowed"/state banner/guidance/ownership-record copy; loading placeholders now read "Loading Settings details".
- Screen title no longer promises "accounts": "Settings | Global preferences, appearance, storage, and app behavior | Local" (matches the actual categories: preferences, appearance, storage, privacy, diagnostics, advanced config).
- Palette: help text for "Settings & Preferences: Open Settings Tab" changed from "Navigate to Tools & Settings tab" to "Navigate to the Settings tab" (both search() and discover() lists); TAB_SETTINGS help in TabNavigationProvider de-"accounts"-ed; the command name itself (pinned by test_command_palette_providers.py) is unchanged.
- Theme Delete guard: the old code lumped shipped ALL_THEMES names into a variable named `custom_names` and blocked everything with "Cannot delete built-in or custom themes". Now built-in themes (textual-dark/light + shipped catalog) get an accurate "'X' is a built-in theme and cannot be deleted", saved custom user themes (TOML files in ~/.config/tldw_cli/themes) are deletable, and deleting a name with no saved file warns "No saved custom theme named 'X'" instead of silently doing nothing.
- New delete-guard tests mount the editor in Tests.textual_test_harness.IsolatedWidgetTestApp rather than a real TldwCli: the real app's startup machinery (initial-screen push with a DestinationHeader, background workers) stays live during run_test and raced the delete path's message-loop pumps, producing an unrelated NoMatches('#workbench-header-title') flake only when run alongside other UI files.

Files changed:
- tldw_chatbook/UI/Screens/settings_screen.py (copy rewords only)
- tldw_chatbook/app.py (palette help text, TAB_SETTINGS help text)
- tldw_chatbook/Widgets/settings_theme_editor.py (delete guard logic + copy)
- Tests/UI/test_settings_configuration_hub.py (pinned copy: "Planned", "Managed in: MCP", "Loading Settings details")
- Tests/UI/test_destination_shells.py, Tests/UI/test_unified_shell_phase6_nielsen_closeout.py (title pin)
- Tests/UI/test_settings_theme_editor.py (3 new delete-guard tests, isolated harness)
- Tests/UI/COMMAND_PALETTE_TESTING.md (manual-test doc destination name)

Tests: Tests/UI/test_settings_configuration_hub.py, test_settings_theme_editor.py, test_settings_category_sweep.py, test_settings_footer_hints.py, test_settings_save_commit_models.py, test_settings_narrow_layout.py (282 passed), plus test_command_palette_providers.py; green except the documented pre-existing destination failures and the TASK-2831 residual race, neither touched here.

Follow-up (code-review fixes, 2026-08-05): (1) Delete-guard vocabulary now matches the theme tree's grouping - textual-dark/light report "built-in theme", shipped ALL_THEMES catalog entries report "shipped theme". (2) Guard is file-existence-first: a user theme file is deletable even when its name shadows a shipped catalog theme (previously stranded by the save guard allowing the shadowing save); regression test added. (3) The 8 domain follow-up values shortened to "... from Settings in a future release." so the "Planned" row label no longer doubles the word. (4) Delete tests now populate the tree pre-delete (tree-removal branch genuinely exercised) and assert the editor falls back to textual-dark. (5) Sidebar/pane category title renamed "Domain Ownership" -> "App Areas (read-only)"; description keeps all 8 domain names so category search still surfaces them (verified by test_settings_sidebar_collapses_read_only_domain_stubs). (6) Dash style kept as spaced hyphen, the file's dominant convention (74 vs 1 occurrences). Re-verified foreground: Tests/UI/{test_settings_theme_editor,test_settings_configuration_hub,test_settings_category_sweep,test_settings_footer_hints,test_settings_narrow_layout}.py - 277 passed.
<!-- SECTION:NOTES:END -->
