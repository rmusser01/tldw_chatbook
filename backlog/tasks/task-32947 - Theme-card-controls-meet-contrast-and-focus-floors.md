---
id: TASK-32947
title: Theme card controls meet contrast and focus floors
status: Done
assignee: ['@claude']
created_date: '2026-09-24 14:45'
labels: [settings, theme, a11y, css]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Settings ▸ Theme card's controls fall below WCAG floors when measured from painted cells at 190x55. The Apply label reads at 3.78:1 on textual-dark and Delete at 4.47:1. The five default-variant buttons (New, Clone, Export, Generate, Set as launch default) are filled with the card colour, so they read as bold prose, and focusing one shifts the fill by only 1.1:1. The shared action-row focus rule swaps Apply's and Delete's colour for neutral, so a focused button loses its meaning. The focused preset swatch draws a `$primary` ring on the swatch at 1.51:1. 22 of 40 presets measure under 3:1 against the dark card (17 on light), so near-card presets read as gaps. On light themes the Dark-theme checkbox's off glyph is painted in the card's own fill (1.0:1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every Theme-card action button label is at least 4.5:1 against its painted fill, at rest and when focused, under textual-dark, textual-light, gruvbox_dark and solarized_light
- [x] #2 Every Theme-card action button has visible edges at least 3:1 against the card, so default buttons no longer read as prose
- [x] #3 A focused button changes fill by at least 3:1, and a focused Apply/Save/Reset/Delete keeps its variant hue
- [x] #4 Every preset swatch has edges at least 3:1 against the card, and the colour still fills the swatch
- [x] #5 A focused swatch's indicator is at least 3:1 against the card beside it and differs visibly from the resting swatch
- [x] #6 The Dark-theme checkbox's off glyph is visible (at least 3:1)
- [x] #7 A painted-segment contrast test pins #1-#6 and fails on the pre-change stylesheet
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Find where the Apply/Delete colours come from: app-wide Button variant styling, or theme-card CSS.
2. Write a painted-segment contrast test on the real Settings destination at 190x55 across four themes, and see it fail.
3. Restyle the card's buttons, swatches and checkbox in `_settings_splash_theme.tcss`, keyed on the existing `theme-editor-action` / `color-preset-swatch` classes.
4. Pin any hue tokens the new styling relies on, and extend the theme-contrast gate to cover them.
5. Rebuild the CSS bundle, then run the suites, ruff and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Root cause (app-wide, Textual).** The filled-variant colours come from Textual's own `Button.-style-default.-primary/-error/...` DEFAULT_CSS, not from theme-card CSS. The label is `$button-color-foreground` = `auto 87%`. Textual picks black or white by *perceived brightness*, not WCAG luminance, and then blends at 87%. So it picks the wrong pole on mid-tone fills: solarized primary `#268bd2` needs black and gets white (3.15:1); gruvbox error `#fa4934` gets 2.93:1. No CSS-only fix works on an arbitrary fill, so this change does not touch filled buttons app-wide. The fix is scoped to the Theme card.

**Theme card (CSS only, `components/_settings_splash_theme.tcss`).**
- Every `.theme-editor-action` is now a bracketed chip on the card: `border-left/right: outer`. Neutral buttons use a `$ds-text-primary` label with `$foreground` brackets. Variant buttons (`:enabled` only, so a disabled Delete still paints disabled) use their readable text tint for both the label and the brackets: `$text-primary`, `$text-success`, `$text-warning`, `$ds-status-error-readable`.
- Focus (`#settings-theme-card .settings-action-row .theme-editor-action:focus`, specificity (1,3,0)) inverts the chip into its hue. It has to beat the Settings sheet's `#settings-shell Button:hover:focus` (1,2,1), which is the rule that actually neutralised the fill. The first attempt at (0,3,0) lost to it. Neutral buttons invert to `$text-primary`, following the app's `$ds-focus-accent` grammar.
- Base rules stay at (0,1,0), so app-wide `Button:hover` and `Button:disabled` still apply to the neutral chips.
- Swatches: `border-left/right: vkey $foreground` draws thin side rules. Border cells keep the swatch fill, so all 3 columns still show the colour and the width is unchanged. Focus drops the `*:focus` outline (which drew `┌─┐` over the colour) and thickens the rules into `outer` half-block brackets, a shape change judged against the card.
- Checkbox: the off glyph now uses `$ds-text-muted` (Textual's dim-X convention). On keeps `$success`, and the On/Off label still states the state in words.

**Theme-wide pin (`css/Themes/themes.py`).** `_READABLE_TEXT_HUES` gains `text-success`, `text-warning` and `text-error`. Before this change, 21 themes failed AA on `text-success` and 27 on `text-warning`, textual-light among them (Save 3.22:1, Reset 2.80:1). On `text-error`, 5 Textual built-ins failed. No app tcss reads `$text-success` or `$text-warning`. `$text-error` feeds `$ds-status-error-readable`. Textual's own consumers are text uses too (syntax highlight tokens, flat buttons, toast markup). The pin only moves a tint that fails AA, and only toward the text pole, so it can only raise contrast. `Tests/UI/test_theme_contrast.py` now gates `text-success` and `text-warning` on every shipped theme.

**Measured (label contrast; neutral = New/Clone/Export/Generate/Set-default):**

| theme | control | before | after (rest / focused) |
|---|---|---|---|
| textual-dark | Apply | 3.78 | 5.12 / 5.25 |
| textual-dark | Delete | 4.47 | 4.62 / 4.77 |
| textual-dark | neutral fill vs card | 1.22 (focus shift 1.12) | brackets 3:1+, focus shift 5.25 |
| textual-light | Save / Reset | 7.84 / 9.12 (fill = card 1.51 / 1.27) | 4.57 / 4.86 rest, 4.83 / 5.14 focused |
| textual-light | Delete | 4.47 | 6.24 / 6.50 |
| gruvbox_dark | Delete | 2.93 | 6.20 / 6.38 |
| solarized_light | Apply | 3.15 | 5.90 / 5.83 |
| solarized_light | Delete | 3.83 | 7.02 / 7.02 |
| all four | focused Apply/Delete hue | lost (neutral) | kept (inverted) |
| all four | preset swatch edge | none (22/40 dark, 17/40 light under 3:1) | `$foreground` rule, ≥3.64:1 |
| textual-dark | swatch focus ring | 1.51 on swatch | `outer` bracket, ≥3:1 vs card |
| light themes | Dark-theme off glyph | 1.0 | ≈5:1 |

**Tests.** `Tests/UI/test_settings_theme_card_contrast.py` is new. It has 3 cases × 4 themes on the real Settings destination with production CSS at 190x55, and it reads painted segments. It fails 12/12 on the pre-change CSS and themes.py, and passes 12/12 after.

**Files:** `tldw_chatbook/css/components/_settings_splash_theme.tcss`, `tldw_chatbook/css/tldw_cli_modular.tcss` (rebuilt), `tldw_chatbook/css/Themes/themes.py`, `Tests/UI/test_settings_theme_card_contrast.py`, `Tests/UI/test_theme_contrast.py`, `Docs/User_Guide/settings.md`.
<!-- SECTION:NOTES:END -->
