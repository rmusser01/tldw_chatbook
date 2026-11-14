---
id: TASK-40542
title: 'TextArea caret is under 3:1 on its cursor line on many themes'
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
A TextArea's caret is painted in `$input-cursor-background` on the widget's `$boost` cursor line, and that pairing is under 3:1 on 47 of the 267 theme x pane rows on dev today (`4b90eb860b`), across 19 themes. No open PR fixes it. Open PR #3468 (TASK-38229) pins the Input caret only; its fill model describes the Input's focus tint, not the TextArea cursor line. As a side effect it lifts 12 TextArea rows on the 4 dark themes it re-pins, which leaves 41 rows under 3:1 once it merges. Found by the Roleplay B2b follow-up burn-down (2026-11-07), measured as painted by #3468's theme x pane probe (its TextArea rows).

Affected themes on dev (19): ancient_papyrus, atom-one-dark, bunny_fluff, catppuccin-latte, cloudy_day, confectionery_bliss, kawaii_candy, litestep, pastel_dreams, retro_mint_chip, rose-pine-dawn, sakura_viewing_picnic, solarized_dark, solarized_light, solarized-dark, solarized-light, sweet_sorbet, tokyo-night and vintage_scroll.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The TextArea caret is at least 3:1 against its cursor-line fill on every measurable theme over each of the three panes, measured as painted
- [ ] #2 Themes that already read keep their painted TextArea caret and the Input carets do not change
- [ ] #3 A painted gate covers every measurable theme and pane and, before the fix, is red on every row then under 3:1
- [ ] #4 No boot-parsed CSS growth
<!-- AC:END -->
