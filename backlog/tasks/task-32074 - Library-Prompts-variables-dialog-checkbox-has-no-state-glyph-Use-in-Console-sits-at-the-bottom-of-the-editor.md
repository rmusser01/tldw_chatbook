---
id: TASK-32074
title: >-
  Library Prompts: variables-dialog checkbox has no state glyph; 'Use in
  Console' sits at the bottom of the editor
status: Done
assignee: []
created_date: '2026-09-08 18:26'
updated_date: '2026-09-08 21:44'
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

AC#2: 'Use in Console' moved out of `#library-prompt-editor-actions` (row 49 of 52 live -- below every field and the whole history region) into the editor HEADER, on its own row directly under Basic/Advanced/Info. Same id, same visibility rule, so the screen's handler and the in-place `_sync_...` (which query by id) are unchanged. The DOM/focus-order pinning test in test_library_prompts_canvas.py was extended rather than deleted.

Fix round 1: it first joined `#library-prompt-mode-controls` as a fourth control. That row is a bare `Horizontal` with no overflow rule anywhere, and the prompts work pane floors at 48 cells: measured on the canvas at 40/48/56 columns, the button starts at column 48 and lands entirely outside a 44-cell canvas (at 56 the label clipped to 'Use in '). It now rides its own `#library-prompt-header-actions` row -- the same ruling task-30043 made on the media canvas when a fourth action clipped that row, and the shape the Media Reader itself uses (its action row sits beside the mode row, not in it). Pinned at 44, 80 and 140 columns.

Files: tldw_chatbook/Widgets/Console/prompt_variables_dialog.py, tldw_chatbook/Widgets/Library/library_prompts_canvas.py, Tests/UI/test_library_crit8_polish_media.py, Tests/UI/test_library_prompts_canvas.py, Docs/User_Guide/library/prompts.md.

Fix round 2 (critique-8 re-review, group polish-media): the header-actions row itself carries `ds-toolbar` (min-height 1, `$ds-surface-raised`), so with the button hidden -- every dirty edit, every new prompt, the conflict state -- the row still painted a full-width empty raised strip under the mode tabs, covering the fields below it. `header_actions.display` now follows `use_console.display` in both compose() and `sync_lifecycle_actions`. Pinned by `test_header_actions_row_collapses_when_use_in_console_is_hidden` in Tests/UI/test_library_crit8_polish_media.py.
<!-- SECTION:NOTES:END -->
