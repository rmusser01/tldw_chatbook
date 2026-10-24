---
id: TASK-36663
title: 'Focus edges drawn with other tokens fall under 3:1 (152 edges)'
status: To Do
assignee: []
created_date: '2026-10-24 16:59'
labels:
  - ui
  - accessibility
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Exists on dev today and was not fixed by TASK-36479, whose scope was the 43 focus edges drawn with $ds-action-focus (they paint $ds-active-fg or $text-accent now). The same sweep (Tests/UI/test_theme_contrast.py, 89 measurable themes, against $surface, $panel and the fill each focus rule paints) found 152 more focus edges that fall under 3:1. They are checked in as per-edge debt in Tests/UI/focus_edge_shortfall_on_dev.tsv, a list the test lets only shrink. Each item below exists on dev today:

- $ds-focus-accent ($primary): 36 edges in 13 sources, under 3:1 in 29 themes (pastel_dreams 1.03:1). Among them is the global *:focus { outline: solid $ds-focus-accent } fallback in core/_reset.tcss, which every widget without its own focus rule paints.
- $ds-input-focus-accent ($primary): 58 edges in 10 sources, 29 themes.
- $ds-input-focus-border ($primary): 10 edges in 6 sources, 29 themes. These are the Input:focus and TextArea:focus frames in components/_forms.tcss and their copies.
- Raw $primary: 7 edges in 5 sources (the nav buttons, the file-notes git panel, the emoji picker search), 29 themes.
- Raw $accent: 37 edges in 21 sources, 37 themes (kawaii_candy 1.02:1).
- $foreground: 2 edges (the theme editor's colour swatches), 1 theme (pastel_dreams 2.19:1 on $panel).
- $tldw-active-edge: 2 edges (the active top tab when focused). Measured against $surface/$panel they fail 48 themes; their real ground is the tab bar, where test_app_chrome_lines_* measures them.
- The generic Button:focus fill. A Button whose own focus rule sets no fill takes $tldw-focus-fill, and a $ds-active-fg edge on it paints 2.33-2.88:1 in tokyo-night, atom-one-dark, catppuccin-frappe and earthy_nature. This covers the Library rail rows and hub recents, the Console rail carriers and the prompts modal's buttons. The sweep does not model fills set by other rules.
- litestep's $background. $ds-active-fg reads 2.12:1 on it, for any focus edge whose outside cell is the screen background.
- The 13 non-focus edges still drawn with $ds-action-focus stay under 3:1 in 29 themes: .is-active, three highlighted picker rows, the destination header and six modal frames.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every focus edge listed in Tests/UI/focus_edge_shortfall_on_dev.tsv clears 3:1 against $surface, $panel and its own rule's fill on every measurable theme
- [ ] #2 The debt list is empty and removed, so the sweep measures every focus edge with no exemption
- [ ] #3 Button focus edges clear 3:1 against the fill they actually paint on, including the generic Button:focus $tldw-focus-fill, on every measurable theme
<!-- AC:END -->
