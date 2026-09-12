---
id: TASK-31983
title: 'Library media: keyboard focus on a list row is invisible against selection'
status: Done
assignee: []
created_date: '2026-09-07 22:48'
updated_date: '2026-09-07 23:48'
labels:
  - library
  - media
  - ux
  - css
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #6 P1 (lead), both assessors. `.library-media-row:focus` and `.library-media-row-selected` share the identical treatment in `_agentic_terminal.tcss` (`background: $ds-focus-bg; color: $ds-focus-fg; text-style: bold underline`), so the row the keyboard is ON is visually indistinguishable from the row that is selected/open. Assessment B measured the focused-row background against the open-item row and found them separated by three units in a single colour channel (rgb(28,70,102) vs rgb(25,68,102)) with bold+underline shared and no glyph, so arrow-key navigation reads as a dead key. In a keyboard-first product this is the sharpest contradiction of the brand. Applies to the sibling row canvases (conversations/notes/prompts) too, which share the same idiom.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A focused list row is visually distinct from a selected/open row at 235x52 and 100x30, with a cue that is not colour-only (a glyph or a clearly separated background, never colour-alone)
- [x] #2 Arrow-key movement between rows is visible without a selection change (a painted pin over the real screen asserts the focused row's cue moves on Down/Up)
- [x] #3 The compact 2-cell row keeps its label intact (no outline: heavy regression)
- [x] #4 The same distinction holds for the conversations, notes and prompts row canvases
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the root cause: `.library-media-row:focus` and `.library-media-row-selected` share the identical treatment. 2. Give every list `:focus` a focus-only border-left cursor bar that `-selected` does not carry; swap padding 0 1 -> 0 1 0 0 so the content column is fixed. 3. Apply to media/trash/notes/notes-folder/prompt + a new conversation :focus. 4. Painted pin at 235x52 and 100x30, red first.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
A focus-only `border-left: thick $ds-action-focus` cursor bar (a filled block) now distinguishes the keyboard-focused row from the selected/open row, which keeps its background-only treatment. Padding `0 1` -> `0 1 0 0` puts the bar in the column the left pad vacated, so the content column is identical blurred vs focused (no jump, no clip) and no `outline: heavy`. Applied to `.library-media-row`/`.library-media-trash-row`, `.library-notes-row`, `.library-notes-folder-row`, `.library-prompt-row`, and a newly-added `.library-conversation-row:focus`. Reuses `$ds-action-focus`; the bar is a shape cue, so it survives a monochrome terminal. Bundle regenerated. Pin `test_library_row_focus_cue_t31983.py` (8) asserts the focused row carries the bar and the selected row does not, at both sizes, and that the cue moves without a selection change. Renumbered from task-31976 after a remote id collision. Files: tldw_chatbook/css/components/_agentic_terminal.tcss, tldw_chatbook/css/screen_agentic_library.tcss (bundle), Tests/UI/test_library_row_focus_cue_t31983.py (new), Docs/User_Guide/library.md.
<!-- SECTION:NOTES:END -->
