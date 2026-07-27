---
id: TASK-875
title: >-
  LabModeStrip's active label is invisible — a one-row strip inheriting a
  three-row border
status: Done
assignee: []
created_date: '2026-07-26 23:20'
updated_date: '2026-07-27 15:06'
labels:
  - ui
  - bug
  - css
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The active mode's label does not render in `LabModeStrip`. The user cannot see which mode is selected.

The mechanism, confirmed while fixing the identical bug in the Watchlists tab strip: the strip pins itself to `height: 1`, while the global `.is-active` rule in `css/components/_agentic_terminal.tcss` applies `border: round $ds-action-focus`. A bordered button is three rows tall, so inside a one-row strip only its top border survives — the row that would have held the text is clipped away.

This is the second occurrence of the bug class. The Watchlists fix (Phase C, task 6) added a strip-scoped rule with enough specificity to beat the global single-class selector — `border: none` plus `text-style: bold underline` — modelled on the MCP mode strip, which does not have the bug. Do not edit the shared `.is-active` rule; other screens depend on the border it draws.

Worth checking whether any other one-row strip inherits the same rule.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The active mode's label is legible in `LabModeStrip`
- [x] #2 The active mode is visually distinguishable from the inactive ones
- [x] #3 The strip stays one row tall
- [x] #4 A test fails when the active label is not visible, asserting against the compositor rather than `render_line()`
- [x] #5 `css/components/_agentic_terminal.tcss`'s shared `.is-active` rule is unchanged
- [x] #6 Every other one-row strip carrying `.is-active` has been checked for the same defect, and any found are listed here or fixed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The underlying bug was already fixed before this task was filed: commit 880febc05 ("fix(lab): render the active mode chip's label", 2026-07-26 11:20) moved LabModeStrip's chip rules app-tier into css/features/_lab.tcss, mirroring .personas-mode-chip.is-active and #mcp-mode-strip's own border-none override -- and added Tests/UI/test_lab_mode_strip.py, which already includes a compositor-based assertion (test_active_mode_chip_label_is_actually_rendered, using screen._compositor.render_strips()) satisfying AC #4. That commit landed roughly 19 hours before the Watchlists tab-strip fix this task cites as its "worked example" (9f486f337), and this task's own renumbering commit (c5c08ec76) landed after both. So no code change was needed for AC #1-#5; verified by re-running Tests/UI/test_lab_mode_strip.py (18/18 passing) against the current tree.

AC #6 sweep (performed fresh): grepped every .is-active consumer across tldw_chatbook/css and its Python call sites. All one-row-strip (height:1) consumers already carry a border-none + background/color override: LabModeStrip (_lab.tcss), MCP mode strip and MCP hub rail rows, Watchlists tab strip, Lab rail rows, Evals rail rows, Personas mode chip and Personas library rows, and the generic .workbench-mode strip (chat_screen.py) and .library-source-action (height:1 via $ds-library-source-action-height). Two other .is-active consumers were checked and found NOT susceptible: #mcp-audit-subview-strip's sub-view buttons are height:auto (the round border just grows the strip, nothing gets clipped -- confirmed by the file's own comment), and .nav-button (main_navigation.py) is a 3-row button by design (height:3) with its own border override already in place. .library-collection-row and the Watchlist tree's tag/root/watchlist/source buttons carry NO custom CSS at all and rely on Button's compact-mode default styling (tree buttons; see task-876) or Button's height:auto default (library-collection-row) -- neither is a one-row STRIP in the sense this bug class requires, so neither was in scope for this AC. No additional occurrences requiring a fix were found.

No files changed for task-875 beyond this task file.
<!-- SECTION:NOTES:END -->
