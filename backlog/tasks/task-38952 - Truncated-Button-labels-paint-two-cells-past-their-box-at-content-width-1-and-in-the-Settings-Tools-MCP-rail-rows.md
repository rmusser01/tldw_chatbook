---
id: TASK-38952
title: "Truncated Button labels paint two cells past their box: at content width 1, and in the Settings ▸ Tools MCP rail rows"
status: To Do
created_date: 2026-11-05 21:35
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
- [ ] #1 A Button with a 1-cell content box paints no wider than its region and the widget beside it keeps its column; the test is red on dev (3 cells painted in 1)
- [ ] #2 In Settings ▸ Tools at 120x30, 100x30 and 80x30 (bundled CSS, rail scrolled to show every row), every mcp-rail-row paints inside its region and the rail's border and scrollbar stay in place; the test is red on dev (12 cells painted in 10)
- [ ] #3 At 211x44 and 235x52, no painted Button label in any destination changes compared with dev, measured with TASK-38740's capture method (cell_len of each rendered line)
<!-- AC:END -->
