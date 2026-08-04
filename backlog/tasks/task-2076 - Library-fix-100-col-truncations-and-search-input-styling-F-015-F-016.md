---
id: TASK-2076
title: 'Library: fix 100-col truncations and search input styling (F-015, F-016)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-04 05:07'
labels:
  - ux-review
  - library
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 100 cols 'Conversations (0)' ellipsizes its count and the search placeholder truncates to 'Search'; the input renders as a borderless black void with stray artifacts. Evidence: library-100x30.png, library_screen.py:15581. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Row counts are never ellipsized at 100 cols,Search placeholder renders fully,Input reads as a field, not a void,Rendered-layout test at 100x30
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (layout/CSS + copy presentation; no schema/boundary changes). Steps: 1. Evidence: read output/ux-review/library-100x30/library-170x50 PNGs in the main checkout; probe live regions at 100x30 (rail width, input inner width, row label clipping) to find why 'Conversations (0)' loses its count and the placeholder truncates to 'Search'. 2. RED tests: rendered-layout test at 100x30 asserting (a) every visible rail row's count suffix survives (count never clipped; title truncates first), (b) the full 'Search Library…' placeholder is visible, (c) the input renders with the app's field treatment (border/background token) not a void. 3. Fix: title-budget/width-aware truncation in library_rail.py so counts are the last thing to shrink; input width/placeholder; #library-search-input field styling in _agentic_terminal.tcss consistent with the app's field-input token; regenerate tldw_cli_modular.tcss via build_css.py; keep check_bundle_sync.py green. 4. Run rail/shell/destination/parity tests + css sync check + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Counts never clip: new LibraryRailRowButton refits its own label on resize (per-button on_resize, so the fit also follows the scrollbar gutter -- the rail's own size doesn't change when its scrollbar appears). Fitting order: F-013 subtitle word-cut with ellipsis (dropped when the leftover space can't teach), then title ellipsizes; the count is never truncated. Label construction centralized in LibraryRail._row_label; _visible_row_title kept as the conversations/media canvases' escape+cap contract via _truncate_row_title. Search input: #library-search-input now takes the sibling-filter field treatment (border tall $ds-grid-line, $ds-surface-raised, padding 0 1) -- a deliberate frame instead of the near-invisible Textual default (which read as a borderless void with stray left-edge border fragments), and the full 'Search Library…' placeholder fits at 100 cols (padding 0 1 reclaims the cells the default 0 2 ate). Bundle regenerated via build_css.py; check_bundle_sync.py green. Tests: two rendered-layout tests at 100x30 in Tests/UI/test_library_shell.py (every row fits with count intact + placeholder/border contract), RED->GREEN; F-013 subtitle pin updated to assert a dim word-cut prefix (the rail's realistic widths rarely fit the full gloss). Verified: full test_library_shell.py 314 passed; destination/parity/multiselect sweep 263 passed + 1 skip (3 failures are the documented pre-existing dev-broken test_library_screen.py ones); rail widget/shell-state/css-sync 35+4 passed; rag/selection/content-hub/smoke 33 passed; ruff clean on changed files (1 pre-existing F401 in test_library_shell.py untouched). Visual: live headless captures at 100x30 and 170x50 (shot_screen.py) confirm the framed field, full placeholder, and 'Convers... (37)' count survival. Note: LibraryRail.__init__'s 'query' attr shadows Widget.query on the instance -- pre-existing collision documented in passing; the final design (per-button resize) avoids in-widget query calls entirely. ADR: not required (layout/CSS presentation; no schema/boundary changes). Commit 28d769df7.
<!-- SECTION:NOTES:END -->
