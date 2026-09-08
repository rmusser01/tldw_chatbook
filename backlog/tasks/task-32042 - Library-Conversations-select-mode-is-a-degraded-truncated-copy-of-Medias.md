---
id: TASK-32042
title: 'Library: Conversations select mode is a degraded, truncated copy of Media''s'
status: Done
assignee: []
created_date: '2026-09-08 14:36'
updated_date: '2026-09-08 17:23'
labels:
  - library
  - conversations
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #7 P1. Conversations' select-mode toolbar renders 'Selec' (truncated 'Select all'), a bare unlabeled marker, and an awkwardly wrapped '0 selected', and its footer omits the space/s hints Media shows. It is the same feature as Media's clean, labelled select mode but inconsistent and less legible. Share the select-toolbar treatment so the two do not diverge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Conversations select mode shows full, untruncated action labels at supported widths, matching Media's grammar
- [x] #2 The select toolbar is built from the shared treatment rather than a divergent per-canvas copy
- [x] #3 Painted pins assert the Conversations select actions are legible (no mid-word truncation) at 235x52 and 100x30
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Conversations' select-mode toolbar packed all four actions into one `ds-toolbar` row whose scoped CSS forced each child to `width:1fr`, so at the narrow conversations list pane they split evenly (~9 cells each) and truncated to `Selec`/`Exp` with `0 selected` wrapping. Fix (Media parity, task-30043 treatment): split into a summary row (count + `Select all N shown`) above a bulk-action row (`#library-conversations-select-actions`: Clear + Export selected), and change the scoped `#library-conversations-canvas > .ds-toolbar > .library-canvas-action` width from `1fr` to `auto` (min-width:0 kept) so the buttons take their content width. The selector is scoped to #library-conversations-canvas, so Media (its own already-`auto` rule) and Notes/Prompts (no matching selector; base .library-canvas-action carries no width) are unaffected. Kept the `library_disabled_action_label(align=True)` idiom, 'Done stays pressable in select mode', and the `actions_disabled`/empty gating. New pin `test_conversations_select_labels_paint_in_full_like_media` asserts the full labels + one-line count at 235x52 and 100x30 (red first: 'Select all 2 shown' clipped to 'Selec'). The crit6 column-hold pin was correctly adapted from a first-glyph to a word-column measurement (`_painted_word_column(..., 'Export')`, the Notes/Prompts sibling standard) — it still asserts the label holds its column across the 0->1 selection flip; first-glyph diverged by design once the label un-clipped (the align pad reserves the `○ ` width). Bundle rebuilt (conversations rules are in the boot bundle). Files: library_conversations_canvas.py, _agentic_terminal.tcss (+ tldw_cli_modular.tcss bundle), Tests/UI/test_library_multiselect_conversations.py, Docs/User_Guide.
<!-- SECTION:NOTES:END -->
