---
id: TASK-38952
title: "Truncated Button labels paint two cells past their box: at content width 1, and in the Settings ▸ Tools MCP rail rows"
status: Done
assignee:
- '@claude'
created_date: 2026-11-05 21:35
updated_date: 2026-11-06 18:35
labels:
- follow-up
- ui
- settings
- textual
priority: low
references:
- tldw_chatbook/__init__.py
- tldw_chatbook/UI/MCP_Modules/mcp_rail.py
- Tests/Architecture/test_zero_width_fold_guard.py
- Tests/UI/test_settings_category_sweep.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05 (load 10-58 during the probes). TASK-38740 (Done) lists both cases below as follow-ups in its notes; nothing filed them.

Both cases have one cause. Textual 8.2.8 (`textual/content.py:676-705`) cuts a wrapped line to the full content width and only then pads it by one cell on each side. So any line wider than its content width minus 2 paints 2 cells past its box. TASK-38740's guard (`tldw_chatbook/__init__.py:122-163`) handles only a fold width of exactly 0. #3408 did not change this.

- **Content width 1.** A probe at 40x6 put `Button("Save")` (`min-width: 0`) next to a `1fr` Static. At width 1 with no padding, the button paints `' S '`, 3 cells, and the Static's row reads `' S XXXX'`: the neighbour is pushed 2 cells right. At width 3 with padding `0 1`, the content box is again 1 cell and the button paints `'  S  '`, 5 cells. Every other content width from 0 to 6 fits.
- **Settings ▸ Tools MCP rail.** In the bundled-CSS Settings harness (`_BundledSettingsHarness`, `Tests/UI/test_settings_category_sweep.py`), whenever the MCP rail shows its 2-cell vertical scrollbar, the built-in source row `mcp-rail-row-1` has a 10-cell content box but paints 12 cells (`' ◦ ⌂ t…     '`). The pane's right border moves 2 cells right, over the scrollbar.
  - Over-paints at: 120x30, 101x30, 100x30, 100x34, 100x38 and 80x30.
  - Fits at: 100x44, 80x44, 70x44, 120x52, 140x52, 160x40, 211x44 and 235x52.
  - So it depends on the rail scrolling, not only on width.
  - The site-level cause is `MCPRail._label_budget` (`tldw_chatbook/UI/MCP_Modules/mcp_rail.py:288-296`), which never goes below 4 cells while `_ROW_CHROME = 8`. The row text is 10 cells (glyph, space, 4-cell name, space, 3-cell count), and the button's side padding leaves only 8.

The fix can go in the shared wrap shim or at each site. The criteria measure both cases.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Button with a 1-cell content box paints no wider than its region and the widget beside it keeps its column; the test is red on dev (3 cells painted in 1)
- [x] #2 In Settings ▸ Tools at 120x30, 100x30 and 80x30 (bundled CSS, rail scrolled to show every row), every mcp-rail-row paints inside its region and the rail's border and scrollbar stay in place; the test is red on dev (12 cells painted in 10)
- [x] #3 At 211x44 and 235x52, no painted Button label in any destination changes compared with dev, measured with TASK-38740's capture method (cell_len of each rendered line)
<!-- AC:END -->

## Implementation Plan

1. Tests first: a 1-cell Button beside a Static (pure Textual app), a wrapped-line width check on `Content._wrap_and_format`, and the scrolled Settings ▸ Tools MCP rail in the bundled-CSS harness at 120x30, 100x30 and 80x30. Show them red on dev.
2. Fix once, in the shared wrap shim (`tldw_chatbook/__init__.py`), not per site.
3. Compare every painted Button label at 211x44 and 235x52 with dev, using TASK-38740's capture probe.

## Implementation Notes

The shared shim now keeps every wrapped line inside its box, so the fix covers both cases and any other site (`tldw_chatbook/__init__.py`, `_install_zero_width_fold_guard`).
- A wrapped box at or below its line-pad (`0 <= width <= 2 * line_pad`, any overflow) renders the shim's one cropped no-wrap line. Before, only `width == 2 * line_pad` with `fold` did, so a 1-cell box painted ``' S '``.
- After upstream formats a wrapped line, a line wider than the box is re-cut: its pad is stripped, the core is truncated to `width - 2 * line_pad` (with an ellipsis when the overflow is `ellipsis`), and the pad is put back. Upstream truncates to the full width before padding. `fold` lines that fit, which are nearly all, are untouched. `MCPRail._label_budget` is unchanged: once the paint is bounded, the row's text shows as `◦ ⌂ t…` inside its 10 cells.
- Pin rewritten on purpose: `test_no_width_raises_and_only_the_zero_fold_width_changes` (`Tests/Architecture/test_zero_width_fold_guard.py`) asserted that every width except the zero fold width rendered exactly as upstream. Wrapped widths 0 to `2 * line_pad - 1` with a line-pad now render the same cropped line, and the test asserts that they fit their box.
- New tests:
  - `test_a_one_cell_label_paints_inside_its_button` (AC#1; `width: 1`, and `width: 3; padding: 0 1`): red `assert 3 <= 1` and `assert 5 <= 3`. The neighbour's `XXXX` is also checked to stay in its column.
  - `test_a_wrapped_line_never_paints_past_its_box` (ellipsis and clip, widths 0-11): red at base, e.g. `('<---', 0, ['  '])`. A probe also showed `<---` painted as ` <--- ` in 4 cells at width 4 on dev.
  - `test_scrolled_mcp_rail_rows_paint_inside_their_rows` (AC#2, `Tests/UI/test_settings_tools_hub.py`, a whole-file UI census entry). It scrolls each row into view and checks the row's rendered lines (`cell_len`) and the screen line's length. Red at base at all three sizes: `('mcp-rail-row-1', 10, [' ◦ ⌂ t…     '])`. With that check disabled at base, the screen line ran past 100 cells (`…│  ◦ ⌂ t…        ││…`), the shifted border. Green after, 3 of 3.
- AC#3: TASK-38740's capture probe (`w7b_probes/test_w7b_labels_tmp.py`: every destination, plus every Settings category, at 211x44 and 235x52; the text of each painted Button cropped to its region). It was run with dev's shim and with the fixed one, all other code the same. 0 of 4,610 painted label texts changed. Four Console rail rows at 235x52 sat one row apart. A second dev-shim capture moved the same four rows the same way, so that is run-to-run noise (session ids normalized).
- Lanes: the architecture file is a target of `pr-fast-lane` already (not a new target; the whole file takes 2.5 s with setup, 0.7 s of tests). The tools hub file is a whole-file census entry. No lane or pin was edited.
- Covering run (run.sh, `-n 4`, load 25-69): 25 UI, Widgets, Watchlists and Wizards files that pin Button paint or line-pad, plus every `Tests/UI/test_mcp_*.py`: 1825 passed, 229 failed. 8 of the 229 were this round's new tests, red because Settings' one-pane mode (TASK-38953) was not in the tree yet. 217 fail at `fu-base-r11` too. The other 4: 2 passed serially, and 2 (`test_trash_page_past_fold_reachable_wide_split_layout`, `test_server_source_empty_tools_diagnosis_uses_refresh_not_disabled_actions`) fail serially at base as well.

Files: `tldw_chatbook/__init__.py`, `Tests/Architecture/test_zero_width_fold_guard.py`, `Tests/UI/test_settings_tools_hub.py`.
