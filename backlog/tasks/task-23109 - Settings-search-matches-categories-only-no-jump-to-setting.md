---
id: TASK-23109
title: Settings search matches categories only - no jump-to-setting
status: Done
assignee: []
created_date: '2026-08-28 14:06'
updated_date: '2026-08-29 02:24'
labels:
  - ux
  - settings
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The / filter matches category names, not individual settings: searching 'theme' yields a coin flip between Theme (the theme-file editor) and Appearance (where the switch-app-theme setting lives), and a setting like 'reduce motion' cannot be found by name at all. P2 from the 2026-08-28 critique (.impeccable/critique/2026-08-28T06-32-49Z__tldw-chatbook-ui-screens-settings-screen-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Searching a setting's label (e.g. 'reduce motion') surfaces that setting, and Enter navigates to its category with the setting focused or visibly highlighted
- [ ] #2 Ambiguous matches disambiguate with scope text (category and group) in the results line
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Setting-level search lives in a new UI/Screens/settings_search_index.py, and Enter lands on the setting itself: the landing expands enclosing Collapsibles and scrolls before focusing (fields inside a shut disclosure are focusable at zero height, so keystrokes went into an off-screen input), and a disabled target opens its category with an explanation instead of silently doing nothing. Coverage grew to include the Agents form, all seven TTS provider forms and the Console remote-images toggle; a drift guard mounts every category and fails on any rendered-but-unindexed setting, with its harness blind spots declared rather than silent. Ranking keeps established landings from being re-routed by newly indexed rows, and the status line's growth is bounded so the category rail does not lose rows mid-search. PR #2170.
<!-- SECTION:NOTES:END -->
