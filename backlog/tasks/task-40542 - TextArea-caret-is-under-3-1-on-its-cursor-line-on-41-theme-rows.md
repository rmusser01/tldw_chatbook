---
id: TASK-40542
title: 'TextArea caret is under 3:1 on its cursor line on 41 theme rows'
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
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A TextArea's caret is painted in `$input-cursor-background` on the widget's `$boost` cursor line, and that pairing is under 3:1 on 41 of the 267 theme x pane rows (47 rows before TASK-38229's change). It exists on dev today (`4b90eb860b`). No open PR fixes it: PR #3468 (TASK-38229) pins the Input caret only, and its fill model describes the Input's focus tint, not the TextArea cursor line. Found by the Roleplay B2b follow-up burn-down (2026-11-07), measured as painted in `fu-caret-report.md` (probe rows with `widget == TextArea`).

Affected themes include ancient_papyrus, atom-one-dark, bunny_fluff, catppuccin-latte, cloudy_day, confectionery_bliss, kawaii_candy, litestep, pastel_dreams, retro_mint_chip, rose-pine-dawn, sakura_viewing_picnic, solarized_dark, solarized_light, solarized-dark, solarized-light, sweet_sorbet, tokyo-night and vintage_scroll.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The TextArea caret is at least 3:1 against its cursor-line fill on every measurable theme over each of the three panes, measured as painted
- [ ] #2 Themes that already read keep their painted TextArea caret and the Input carets do not change
- [ ] #3 A gate covers all measurable themes and is red on dev for the rows above
- [ ] #4 No boot-parsed CSS growth
<!-- AC:END -->
