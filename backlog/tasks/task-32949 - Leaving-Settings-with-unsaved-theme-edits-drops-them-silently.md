---
id: TASK-32949
title: Leaving Settings with unsaved theme edits drops them silently
status: Done
assignee:
  - '@claude'
created_date: '2026-09-24 22:30'
labels:
  - settings
  - theme
  - ux
priority: medium
dependencies:
  - TASK-32941
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-32941 added a Stay / Discard / Save prompt when leaving the Theme category with unsaved edits, but only for category switches inside Settings. Navigating away from the Settings screen itself (tab bar, command palette, Ctrl+digit routes) still discards the in-progress theme silently, exactly the data loss 32941 closed for the in-screen path. Found by the 2026-09-24 branch review of `fix/theme-ux-wave`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Leaving the Settings screen by any navigation route while the theme editor has unsaved edits shows the same Stay / Discard / Save choice as a category switch
- [x] #2 Stay keeps the user on Settings ▸ Theme with the edit intact; Discard leaves and drops it; Save saves then leaves (a refused save or pending overwrite keeps the user on Theme)
- [x] #3 Navigation away from Settings with no unsaved theme edits is unchanged (no prompt, no added latency)
- [x] #4 Pilot tests cover each route used and each choice
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Done as part of the Qodo review fixes on `feat/theme-picker-pr3` (Qodo 4100998925).

- **Approach:** `SettingsScreen.confirm_navigation()` implements the app's existing leave hook. Every route (tab bar, command palette, shortcuts, in-screen links) posts `NavigateToScreen`, and `TldwCli.handle_screen_navigation` awaits the outgoing screen's `confirm_navigation` inside its navigation worker (the seam TASK-1143 added; Personas uses it the same way). With a modified theme editor it pushes the existing `ThemeLeaveModal` (TASK-32941) via `push_screen_wait`. Stay returns False. Discard returns True. Save runs `on_save_theme()` and returns `not is_modified`, so a refused save or a pending overwrite confirmation keeps the user on Theme.
- **No edits, no prompt:** without an editor or with `is_modified` False the hook returns True straight away. It does no I/O and pushes no modal.
- **Related:** a name-only edit now marks the editor modified (Qodo 4100998919), so it is covered too.
- **Tests (`Tests/UI/test_settings_theme_picker_screen.py`):** there is one pilot test per choice (no edits, Stay, Discard, Save, and refused plus pending Save), driven through the hook the app awaits. That the app awaits this hook for every route is already pinned route-agnostically in `Tests/UI/test_screen_navigation.py` (`test_navigation_confirms_with_outgoing_screen_and_honors_veto`). There is no separate pilot per entry surface, because all of them converge on that one handler. Those navigation tests are CI-only in a clean worktree (`RecoveryRequired`, identical failure set at HEAD).
- **Files:** `tldw_chatbook/UI/Screens/settings_screen.py`, `Tests/UI/test_settings_theme_picker_screen.py`, `Docs/User_Guide/settings.md`.
<!-- SECTION:NOTES:END -->
