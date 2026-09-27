---
id: TASK-33062
title: Theme picker keys are discoverable and help copy matches the controls
status: Done
assignee:
  - '@claude'
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P2. The list's keys (Enter, t, c, n, e, r, Delete, i) work but appear nowhere; F1 says no category shortcuts exist and three places (F1, the ownership boundary copy, the inspector Save row) name an Apply button that no longer exists. Spec §5 promised the keys in the footer. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The footer and F1 help for Theme list the picker's keys
- [x] #2 No Settings copy refers to an editor Apply button; the copy describes Use/Try and the editor's Save
- [x] #3 Tests that pinned the stale 'Apply/Save/Reset' copy are updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. One THEME_SHORTCUTS tuple on SettingsScreen (Enter use, t try, c clone, n new, i import, e/r/Del for your themes).
2. F1: _category_footer_shortcuts returns it for THEME (so the "No shortcut keys" note goes away) plus one note that the keys act while the theme list has focus.
3. Footer: _footer_shortcut_entries appends it while the theme list has focus (the keys are the list's own bindings; ADR-031 rule 4 advertises only keys that work).
4. Rewrite the ownership boundary copy and the inspector Save row to name Use/Try and the editor's Save; update the pinning test in test_settings_save_commit_models.py.
5. Tests: footer with list focus, F1 shortcuts + note, no "Apply" in Theme copy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
New `SettingsScreen.THEME_SHORTCUTS` (Enter use, t try, c clone, n new, i import, e/r/Del for your themes). `_category_footer_shortcuts(THEME)` returns it, so F1 lists the keys and drops "No shortcut keys…". F1 also says the keys act while the theme list has focus. The footer appends them only while a `ThemeOptionList` has focus (ADR-031 rule 4). In the filter they would type and in the editor they do nothing, so they are hidden there. A guarded `_theme_list_focused()` keeps bare-screen tests safe. The ownership boundary copy and the inspector Save row now say "Use/Try switch themes; the editor's Save stores a (theme) file".

Tests: the pinning test in test_settings_save_commit_models.py was updated and now runs under @private_profile_test, since it was an ADR-126 environment red without it. test_settings_footer_hints' pure mapping test now expects THEME's list keys. Two new Pilot tests cover the footer (list focused vs filter focused) and F1.

Full-screen check: the footer shows all 8 keys at 211x44 and 235x52, and the first 4 at 80x24 (the footer's own narrow collapse). F1 always has all of them.
<!-- SECTION:NOTES:END -->
