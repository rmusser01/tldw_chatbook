---
id: TASK-31250
title: Theme editor - saved user themes are never loaded, so Save cannot make a theme
  the launch theme
status: To Do
created_date: 2026-09-04 05:23
assignee:
- '@claude'
labels:
- ui
- settings
- theme-editor
- ux-review-2026-09
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The editor's Save writes ~/.config/tldw_cli/themes/<name>.toml, but the only code that ever reads that directory is the editor widget itself. app.py registers ALL_THEMES at startup, Appearance's Theme options and the palette's 'Theme: Switch to' list shipped themes only. Setting general.default_theme to a saved theme silently falls back to textual-dark while Appearance displays 'Current: <name>'. The Save hint ('stores the theme for future sessions') and Docs/User_Guide/settings.md (Interface - Theme, Quirks) promise the opposite. Reproduced live. Evidence: live walkthrough of origin/dev 59d987015d on 2026-09-03 (isolated profile, tmux 235x52) plus a dual-agent impeccable critique; snapshot .impeccable/critique/2026-09-04T04-45-47Z__tldw-chatbook-widgets-settings-theme-editor-py.md. Heuristic score 17/40.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every readable *.toml under the active profile's themes directory is registered as an app theme at startup; an unreadable file is skipped with a logged warning and does not block startup
- [ ] #2 A theme saved in the editor is registered immediately after Save (no restart) and can be selected in Appearance > Theme and in the command palette 'Theme: Switch to' list
- [ ] #3 Launching with general.default_theme set to a saved theme applies that theme
- [ ] #4 The editor offers a way to set the current saved theme as the launch default (writes general.default_theme) and the Save hint states what Save does and does not do
- [ ] #5 Docs/User_Guide/settings.md Theme section and the 'theme didn't change after saving' quirk describe the new behaviour, with the Verified-against stamp updated
- [ ] #6 Tests cover: startup registration from a temp themes dir, skip-on-bad-file, Appearance options include the saved theme
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
