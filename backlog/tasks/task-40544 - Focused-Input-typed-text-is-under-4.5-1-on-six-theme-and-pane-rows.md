---
id: TASK-40544
title: 'Focused Input typed text is under 4.5:1 on six theme and pane rows'
status: To Do
assignee: []
created_date: '2026-11-12 09:00'
labels:
  - follow-up
  - contrast-fix
dependencies: []
references:
  - tldw_chatbook/css/Themes/themes.py
  - Tests/UI/test_theme_contrast.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Typed text in a focused Input (`$text` on the house focus tint `$ds-input-focus-bg`, over the pane) is under 4.5:1 on six rows: nord on `$panel` 4.47, litestep on `$background` 3.47, catppuccin-mocha on `$panel` 4.19, catppuccin-macchiato on `$panel` 4.47, catppuccin-frappe on `$panel` 4.12 and atom-one-dark on `$panel` 4.33. It exists on dev today (`4b90eb860b`). No open PR fixes it: PR #3468 (TASK-38229) holds these six rows at their dev value in its painted gate on purpose (`TYPED_TEXT_ON_DEV`). Found by the Roleplay B2b follow-up burn-down (2026-11-07); evidence in `fu-caret-report.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Typed text in a focused Input reads at 4.5:1 or better on every measurable theme over each of the three panes, measured as painted
- [ ] #2 No other theme's typed text or caret changes
- [ ] #3 The six-row allowance in the painted gate is removed
- [ ] #4 No boot-parsed CSS growth
<!-- AC:END -->
