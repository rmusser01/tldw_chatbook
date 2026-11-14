---
id: TASK-38229
title: 'Textual Input caret is under 3:1 on some themes'
status: Done
assignee: []
created_date: '2026-10-30 18:43'
labels:
  - a11y
  - css
dependencies: []
references:
  - Tests/UI/test_theme_contrast.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Textual's own caret colours over a letter measured 2.53:1 on pastel_dreams (Roleplay frame B2b's contrast work). It exists on dev today for every Input; the Workflows search restyles its caret, and B2b does not fix the others. Found during Roleplay frame B2b (TASK-33910.4, plan Docs/superpowers/plans/2026-10-26-roleplay-b2b-toolbar-new-import.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every Input's caret clears 3:1 against its fill on every measurable theme, pinned by a painted test
- [x] #2 No Input's typed text drops below 4.5:1 as a result
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-measure on dev: paint a plain Input on $surface, $panel and $background on every measurable theme and read the caret's block, the letter under it and the typed text.
2. Pin the caret per theme in `ensure_readable_text_hues` (zero CSS bytes: `input-cursor-background` / `input-cursor-foreground` are generated names no tcss defines), only where the one the theme paints fails.
3. Gate it in `Tests/UI/test_theme_contrast.py`: a resolved-value gate per theme and a painted gate over 89 themes x 3 panes; show both red on dev and under named mutations.
4. Paired gate, ruff, boot-CSS and pre-import census on both arms, preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Owner decision 2026-11-05: "Per-theme caret vars (Rec.)", zero boot-CSS growth.

**Present on dev (8ae979d81b), as painted.** The caret block (Textual's `$input-cursor-background`, defaulting to `$foreground`) over a focused field's fill (`components/_forms.tcss` `Input:focus`: `$ds-input-focus-bg`, the house tint over the pane) was under 3:1 on 7 of the 89 measurable themes: pastel_dreams 1.77-2.17:1, atom-one-dark 2.41:1 ($panel), bunny_fluff 2.41-2.74:1, solarized_light 2.63-2.91:1, tokyo-night 2.73:1 ($panel), solarized_dark and the built-in solarized-dark 2.82:1. The letter under the caret (`$input-cursor-foreground`, defaulting to `$background`) was under 4.5:1 on the block on 4: pastel_dreams 2.53:1 (the task's number), bunny_fluff 3.39:1, solarized_light 4.13:1, sweet_sorbet 4.29:1.

**Fix.** `themes.py`: `CARET_VARIABLES` and `_caret()`, called from `ensure_readable_text_hues`. The fill model is Textual's own composition (`pane + tint.tint($foreground 5%)`, textual/widget.py `_get_styles_background`), checked against the painted fills of all 267 rows with 0 misses. The block is kept where it clears 3:1 on all three pane fills, else blended toward white or black, whichever first clears; the letter's ink is kept where it reads 4.5:1 on the block, else blended toward the pole that reads better. A theme that already passes keeps Textual's (or its own) caret: only the 7 + 1 themes above get an entry. No CSS changed; boot-parsed CSS is 598,293 B on both arms.

**Tests** (`Tests/UI/test_theme_contrast.py`): `test_resolved_caret_clears_contrast_on_every_theme` (89 cases, red on dev for the 8 themes) and `test_input_caret_paints_at_non_text_contrast_on_every_theme` (painted, 89 themes x 3 panes; red on dev with 28 caret shortfall lines). AC#2 is held as painted: typed text paints on the field's fill at 4.5:1, or, on the six (theme, pane) rows dev already has under that (`TYPED_TEXT_ON_DEV`: nord, litestep, three catppuccins, atom-one-dark, 3.47-4.47:1, `$text` on the house focus tint), at its dev value. That typed-text shortfall exists on dev today and this task does not fix it.

**Out of scope, exists on dev, not fixed here:** TextArea's caret sits on its `$boost` cursor line and is under 3:1 on 47 theme x pane rows on dev (41 after this change, which moves it only on the re-pinned themes); a focused Input's selected text (`$input-selection-*`) is under 4.5:1 on most themes (camono 2.49:1).
<!-- SECTION:NOTES:END -->
