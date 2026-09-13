---
id: TASK-31283
title: 'Legacy themes: generated text-error/text-muted below AA on their own surfaces'
status: Done
assignee: []
created_date: '2026-09-04 19:03'
updated_date: '2026-09-04 19:09'
labels:
  - themes
  - a11y
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measured during task-31264 across all 70 registered themes: ~31 legacy themes' generated text-error and ~35's text-muted resolve below 4.5:1 against their own surface/panel (worst: earthy_nature text-error 1.7:1, modern_dark_dracula text-muted 1.94:1). These now feed $ds-status-error-readable/$ds-text-placeholder etc., so muted/error text in those themes is hard to read. The 12 Orb themes are gated by test_theme_contrast.py; legacy themes are not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every registered theme's resolved text-error and text-muted (theme variables dict over generated, mirroring runtime resolution) clear 4.5:1 against its surface and panel
- [x] #2 The contrast gate in test_theme_contrast.py covers all registered themes, not just the Orb 12
- [x] #3 Fixes preserve each theme's overall look (base palettes unchanged unless unavoidable)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Mirror runtime resolution ({**generated, **theme.variables}) in test_theme_contrast.py and extend the AA gate to all registered themes (red = worklist)\n2. Solver: hue-preserving lightness walk per failing theme for text-muted/text-error until 4.5:1 vs surface+panel\n3. Apply as per-theme variables dict entries (base palettes untouched); verify all-themes gate green
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
43 legacy themes fixed via per-theme variables dict entries for text-muted/text-error (hue-preserving lightness walks to >=4.55:1 vs surface+panel; base palettes untouched; existing entries' comments preserved). Mechanism: these generated-token names are defined in no tcss source, so dict entries genuinely win at runtime ({**generated, **theme.variables}) — unlike ds-* tokens (task-31264 lesson). Gate extended: test_resolved_readable_tokens_clear_aa_on_every_theme now parametrizes ALL registered themes and mirrors runtime resolution (incl. 'auto NN%' values). Solver kept at scratchpad-only; values are data in themes.py.
<!-- SECTION:NOTES:END -->
