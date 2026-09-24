---
id: TASK-32940
title: Theme editor keeps shipped themes' extra colour variables on Apply/Save/Clone
status: Done
created_date: 2026-09-24 14:45
assignee:
- '@claude'
labels:
- settings
- theme
- accessibility
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every shipped theme carries extra colour variables beyond its ten base colours; they are the contrast (AA) fixes such as muted text, error text, footer keys and input selection. The Settings theme editor only knew the ten base colours, so applying a shipped theme from the editor, or cloning and saving it, silently produced a copy without those fixes: the applied theme and the saved file were less readable than the theme the user picked.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cloning a shipped theme and saving it writes the theme's extra variables, and they come back when the file is loaded at startup and in the editor
- [x] #2 Applying an edited or saved theme applies its extra variables too
- [x] #3 Applying a catalog theme without edits selects the catalog theme by name and registers no copy
- [x] #4 Export includes the same variables as Save
<!-- AC:END -->

## Implementation Plan

1. Failing tests: clone apricot -> save -> reload (startup loader + editor) -> apply keeps variables; unmodified apply selects `apricot`.
2. Carry the loaded theme's variables in editor state (dropping the two per-palette text tints the theme factory re-derives).
3. Write/read a `[variables]` table on Save/Export and in the startup loader; include variables on Apply and the post-Save registration.
4. Unmodified catalog Apply sets `app.theme` to the catalog name.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`SettingsThemeEditor` now holds `_theme_variables` (from the catalog Theme's `variables`, or a user file's `[variables]` table) and `_loaded_catalog_theme`. `_theme_dict()` / `_theme_file_data()` build the Apply/registration dict and the Save/Export TOML, so the three paths cannot drift again. `text-primary`/`text-accent` are not carried: `create_theme_from_dict` re-derives them for the edited palette via `ensure_readable_text_hues`, and no shipped theme declares them explicitly. `load_user_themes` reads the `[variables]` table. Apply with no edits on a loaded catalog theme selects it by name.

Known ceiling: carried variables are the source palette's; heavily re-colouring a clone (e.g. Generate from Primary, or flipping Dark) keeps them unchanged. Not in scope here.

Files: `tldw_chatbook/Widgets/settings_theme_editor.py`, `tldw_chatbook/css/Themes/themes.py`, `Tests/UI/test_settings_theme_editor.py`, `Docs/User_Guide/settings.md`.
<!-- SECTION:NOTES:END -->
