---
id: TASK-32087
title: >-
  Library media: a regional-indicator flag keyword drifts the reader Info
  Keywords line +2 cells
status: Done
assignee: []
created_date: '2026-09-08 20:48'
updated_date: '2026-09-08 22:01'
labels:
  - library
  - media
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #8 gap in task-32044. task-32044 made the LIST-ROW keyword-reason suffix flag-pair-aware (never paints a half-flag), but the reader's Info tab renders the raw keyword on its 'Keywords:' line, and a regional-indicator flag pair there still drifts the row frame +2 cells (line measures 237 vs 235; the right border lands at col 236 vs 234). The flag-pair width miscount lives on this second surface, uncovered by 32044.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A keyword containing a regional-indicator flag emoji does not drift the reader Info 'Keywords:' line frame width or overpaint neighbouring content, at 235x52 and 100x30
- [x] #2 The Info 'Keywords:' rendering uses the same flag-pair-aware width handling task-32044 applied to the list suffix (or an equivalent), rather than mis-measuring the pair
- [x] #3 A pin paints a flag-keyword Info line and asserts the frame stays aligned
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
task-32044 made the LIST-ROW keyword suffix flag-pair-aware but the reader Info 'Keywords:' line rendered the raw keyword. RULING (verified at review): a WHOLE regional-indicator flag PAIR does NOT drift — rich AND Textual measure it as 2 cells (cell_len('🇯🇵')==2; Content.cell_length matches), matching a compliant terminal, so the frame does not drift on a whole pair; critique #8's inferred '+2 drift' was a LONE half-flag or a Textual `Content.wrap(overflow=fold)` split of a pair mid-grapheme at narrow widths. Fixed the universal case: `_display_keywords_text` (library_media_viewer_state.py) drops a dangling trailing regional indicator on the Info line (reusing task-32044's `_trailing_regional_indicators`), keeping whole pairs; the EDIT field keeps the raw keyword (saves round-trip verbatim). LEDGERED out of scope: Textual's fold-wrap splitting a whole pair mid-grapheme at a fold is a framework segmentation limit, not dependency-free fixable in this seam (needs nowrap/ellipsis or an upstream fix). Pin `test_info_keywords_line_drops_a_lone_half_flag_but_keeps_whole_pairs`, red first. Files: library_media_viewer_state.py, Tests/Library/test_library_media_viewer_state.py.
<!-- SECTION:NOTES:END -->
