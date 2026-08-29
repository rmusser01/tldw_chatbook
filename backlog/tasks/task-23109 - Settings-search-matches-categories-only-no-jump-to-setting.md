---
id: TASK-23109
title: Settings search matches categories only - no jump-to-setting
status: To Do
assignee: []
created_date: '2026-08-28 14:06'
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
