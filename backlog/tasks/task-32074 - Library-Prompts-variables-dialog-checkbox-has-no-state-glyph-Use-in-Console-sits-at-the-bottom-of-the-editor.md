---
id: TASK-32074
title: >-
  Library Prompts: variables-dialog checkbox has no state glyph; 'Use in
  Console' sits at the bottom of the editor
status: Done
assignee: []
created_date: '2026-09-08 18:26'
updated_date: '2026-09-08 19:25'
labels:
  - library
  - prompts
  - ux
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The prompt-variables dialog renders its checkbox as an empty box with no checked/unchecked glyph, and 'Use in Console' is at row 49 of 52 in the editor while Media places the equivalent action in the reader header. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 25.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The variables checkbox shows a text-labelled or glyph state
- [x] #2 'Use in Console' is placed consistently with the Media reader's 'Use in Console'
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: the variables checkbox paints an empty frame (Textual paints its inner X in the button's own background while off -- state by colour alone); 'Use in Console' sits in the editor's bottom action strip.
2. Give the dialog's checkbox a glyph button (☑/☐, [x]/[ ] under ASCII glyphs) by overriding ToggleButton._button.
3. Move 'Use in Console' into #library-prompt-mode-controls beside Basic/Advanced/Info, matching the Media reader's header placement; extend the DOM-order pinning test.
4. GREEN; docs stamp on library/prompts.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1: the Prompt-variables dialog's checkbox is now a `GlyphCheckbox` -- a four-line `Checkbox` subclass overriding `_button` to paint ☑/☐ (or [x]/[ ] under `ascii_glyph_mode()`). Textual's ToggleButton renders its inner 'X' in the button's OWN background while the value is False, so an unchecked box is an empty frame and the state is carried by colour alone; that is what the critique met, and it is invisible in a plain-text capture. The widget, its id and its `Changed` messages are untouched, so the dialog's handler and the 'Off'/'On' state Static beside it keep working.

AC#2: 'Use in Console' moved from `#library-prompt-editor-actions` (row 49 of 52 live -- below every field and the whole history region) into `#library-prompt-mode-controls`, beside Basic/Advanced/Info, which is where the Media Reader keeps the same action. Same id, same visibility rule, so the screen's handler and the in-place `_sync_...` (which queries by id) are unchanged. The DOM/focus-order pinning test in test_library_prompts_canvas.py was extended rather than deleted: it now asserts the action is absent from the bottom strip and painted in the header.

Files: tldw_chatbook/Widgets/Console/prompt_variables_dialog.py, tldw_chatbook/Widgets/Library/library_prompts_canvas.py, Tests/UI/test_library_crit8_polish_media.py, Tests/UI/test_library_prompts_canvas.py, Docs/User_Guide/library/prompts.md.
<!-- SECTION:NOTES:END -->
