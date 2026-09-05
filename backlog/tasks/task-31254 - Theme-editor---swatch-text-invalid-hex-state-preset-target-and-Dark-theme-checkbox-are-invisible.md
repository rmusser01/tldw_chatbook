---
id: TASK-31254
title: Theme editor - swatch text, invalid-hex state, preset target and Dark-theme
  checkbox are invisible
status: Done
created_date: 2026-09-04 05:23
assignee:
- '@claude'
labels:
- ui
- settings
- theme-editor
- ux-review-2026-09
priority: high
updated_date: 2026-09-04 06:06
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four states the code sets are never painted: .color-swatch and .color-preset-swatch are 1 row tall with a solid border, so only the top border glyphs render (hex text and the 'Invalid' label never show; presets are '[]' glyphs over a tint); .invalid-color and .selected have no rule in any stylesheet (Settings' convention is settings-invalid-input); the Dark theme Checkbox sits in a 1-row settings-input-row and is clipped to its top border, so its checked state cannot be read (Appearance's Switches are clipped the same way). Live: typing #GGGGGG turned the swatch black and nothing else changed. Evidence: live walkthrough of origin/dev 59d987015d on 2026-09-03 (isolated profile, tmux 235x52) plus a dual-agent impeccable critique; snapshot .impeccable/critique/2026-09-04T04-45-47Z__tldw-chatbook-widgets-settings-theme-editor-py.md. Heuristic score 17/40.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each colour swatch shows its colour and hex text at the row's height
- [x] #2 An invalid hex value shows a visible invalid state on the input using the Settings invalid-input convention and the swatch does not silently turn black
- [x] #3 The colour row a preset will fill is named in visible text (not only a CSS class)
- [x] #4 The Dark theme toggle's on/off state is readable in the row; the same fix or a shared rule covers Appearance's toggle rows
- [x] #5 CSS bundle rebuilt via build_css.py and both bundle files committed; a compositor-text test asserts the swatch hex and the toggle state are painted
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Swatches lose their border (a 1-row bordered Static has no content row) and show the hex on the colour; preset swatches keep a focus outline; invalid input uses Settings' settings-invalid-input class and the swatch reads 'invalid' instead of turning black; the Dark toggle is one row with an On/Off text label (TASK-18960 escape idiom); the preset target is a visible 'Presets fill' Select. Bundle rebuilt. Render tests load the bundle plus screen_agentic_settings.tcss (TASK-25812 split) in Tests/UI/test_settings_theme_editor_render.py.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
