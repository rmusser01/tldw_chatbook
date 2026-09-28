---
id: TASK-32946
title: Theme editor copy fixes
status: Done
created_date: 2026-09-24 12:00
assignee:
- '@claude'
labels:
- settings
- theme
- copy
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Live Preview's accent row was meant to read "[ Send ]   Ctrl+P palette", but the Static parsed "[ Send ]" as markup and it rendered as "    Ctrl+P palette". An invalid colour showed "invalid" on one code path and "Invalid" on the other, with no hint about the expected format.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The preview accent row renders "[ Send ]" literally
- [x] #2 Both invalid-colour paths show the same text with a format hint ("Invalid — use #RRGGBB")
- [x] #3 Decision recorded on committing a partial 3-digit hex while typing
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. `markup=False` on the preview row Statics
2. One module constant for the invalid swatch text, used by both paths
3. Evaluate deferring 3-digit commits to blur
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Preview rows are now `Static(..., markup=False)`. `_INVALID_COLOUR_TEXT = "Invalid — use #RRGGBB"` replaces both "invalid" and "Invalid".

Skipped (AC#3): not committing `#D94` while the user is on the way to `#D94A2B`. `#RGB` is valid CSS shorthand, and `_validate_color_input` deliberately accepts it. Deferring the commit to blur or 7 characters would stop the preview following each keystroke (TASK-31259 made that its contract) and would need a blur/submit commit path. The intermediate commit does no harm: the next keystroke overwrites it, and `is_modified` is correct either way. Worth a design call only if someone asks for it.

Tests: `test_settings_theme_editor_preview_accent_row_renders_brackets`, `test_settings_theme_editor_invalid_colour_names_the_format`.

Files: `tldw_chatbook/Widgets/settings_theme_editor.py`, `Tests/UI/test_settings_theme_editor.py`, `Docs/User_Guide/settings.md`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
