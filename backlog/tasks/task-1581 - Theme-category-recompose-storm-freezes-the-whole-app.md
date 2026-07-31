---
id: task-1581
title: 'Theme category recompose storm freezes the whole app'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-31'
labels:
  - settings
  - theme
  - bug
  - rescore-p1
dependencies: []
priority: high
---

## Description (the why)

Opening Settings → Theme froze the ENTIRE app: every keystroke and mouse
click (including nav tabs) dead, process spinning at ~60% CPU until killed.
Found live during the critique rescore capture ("Theme traps keyboard
navigation"); py-spy-less diagnosis via `sample`, SIGABRT faulthandler dump,
and a temp compose probe. Mechanism: `SettingsThemeEditor.is_modified` is a
watched reactive whose init-time watch call posts ThemeModifiedStatus(False)
on every mount, and the editor's programmatic `load_theme` Input writes
queue Input.Changed events that flip it True — so the screen's
`theme_editor_modified` (recompose=True) oscillated False/True forever, each
flip recomposing the screen and mounting a fresh editor that emitted the
next pair. Pre-existing since the Theme category was wired in (a94f84a94),
not introduced by the remediation PRs.

## Acceptance Criteria (the what)

- [x] Opening the Theme category converges: the mounted editor instance
      survives settling (no recompose storm)
- [x] A clean editor mount emits no ThemeModifiedStatus posts and leaves
      is_modified False
- [x] Real user edits (color value change, dark-mode toggle) still mark the
      editor modified
- [x] Live verification: Theme opens at idle CPU, the category filter and
      all navigation remain usable, and leaving Theme via the filter works

## Implementation Plan (the how)

1. Reproduce live (textual-serve + Playwright, then tmux), capture the spin
   with `sample`/faulthandler, isolate the oscillating reactive with a
   compose-entry probe.
2. RED tests: editor-level (no posts on mount, is_modified stays False,
   user edits still flip it) and screen-level (editor instance identity
   survives settling).
3. Fix in the editor: `is_modified = reactive(False, init=False)`; treat
   programmatic-load no-op Changed events (value equal to stored data) as
   non-edits; same guard for the dark-mode switch sync.
4. GREEN + full affected suites + live tmux verification.

## Implementation Notes

Three-line-class fix in `settings_theme_editor.py`: init=False on the
reactive (kills the False half of the oscillation), an equality guard in
`on_color_input_changed` (load_theme sets `current_theme_data` before
writing Input values, so reload events arrive value-equal and are skipped),
and an equality guard in `on_dark_mode_changed`. No screen-side change
needed — with honest posts the recompose reactive converges. Also fixes the
cosmetic lie where merely opening Theme could report unsaved theme changes.
Live-verified: CPU 0.0% with Theme open (was 56-62%), Theme → Storage →
Theme → Console Behavior round-trips via the filter, all input responsive.
The screen-level regression test pins editor identity across 12 settle
pauses. Files: `tldw_chatbook/Widgets/settings_theme_editor.py`,
`Tests/UI/test_settings_theme_editor.py`,
`Tests/UI/test_settings_configuration_hub.py`.
