---
id: TASK-32945
title: Theme lists tell the truth about origin
status: Done
created_date: 2026-09-24 12:00
assignee:
- '@claude'
labels:
- settings
- theme
- copy
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Appearance ▸ Theme labelled every registered theme that was not in the shipped catalog "(saved)". That included Textual's own themes, so a profile with no saved themes showed "Catppuccin Macchiato (saved)", and "Solarized Light (saved)" sat next to the shipped "Solarized Light". The Theme editor's Built-in group listed only textual-dark and textual-light, hiding the rest of Textual's themes. An empty "Your themes" group showed nothing at all.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 "(saved)" appears only for themes with a file in the user themes directory
- [x] #2 Textual's built-in themes are labelled "(Textual)"; option labels are unique
- [x] #3 The editor's Built-in group lists all of Textual's BUILTIN_THEMES
- [x] #4 An empty "Your themes" group shows a "(none yet)" leaf that does nothing when selected
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. `_appearance_theme_options`: check `_theme_save_target()/<name>.toml` for "(saved)", `textual.theme.BUILTIN_THEMES` for "(Textual)", suffix the id on a label collision
2. Editor tree: iterate BUILTIN_THEMES; add a data-less "(none yet)" leaf when no user files load
3. Widen the editor's built-in-name checks (delete guard, catalog check) to BUILTIN_THEMES so newly listed themes behave like textual-dark
4. Extend the stub test with catppuccin-macchiato/solarized-light; editor tests for the tree
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Appearance: a registered theme gets "(saved)" only when `<themes dir>/<name>.toml` is a file, otherwise "(Textual)" if it is in `BUILTIN_THEMES`, otherwise a plain title. A label that collides with an earlier one gets " · <id>". Editor: the Built-in node lists `BUILTIN_THEMES`. `_is_catalog_theme` and the Delete guard use the same set, so Reset and Set-as-default treat e.g. catppuccin-mocha as catalog and Delete calls it built-in. The Save guard and the Name-box disable still name only textual-dark/light: the harden lane owns Save, and shipped names are already allowed to be shadowed by a saved file. "(none yet)" carries no `data`, so `on_theme_selected` ignores it.

Tests: `test_settings_appearance_theme_options_include_registered_user_themes` (gate-free; monkeypatches `_theme_save_target`), `test_settings_theme_editor_tree_lists_every_textual_builtin`, `test_settings_theme_editor_empty_your_themes_says_none_yet`.

Files: `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/Widgets/settings_theme_editor.py`, both test files, `Docs/User_Guide/settings.md`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
