---
id: TASK-33074
title: Long theme names do not wrap the colour strip at 80x24
status: Done
assignee: []
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P3. At 80x24 long names push the colour strip onto a second line. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 80x24 each theme row stays on one line, with long names truncated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. _row takes an optional width and truncates only the name (Rich Text.truncate, ellipsis), keeping the colour strip and markers.
2. The picker computes the width from the list's content region (minus the scrollbar and option padding) and refits the prompts in place when the list resizes.
3. Test at 80x24 on the real Settings screen with a long-named theme: every theme row is one line and the long name ends with an ellipsis.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
_row takes an optional width and truncates only the name (Rich Text.truncate, ellipsis) so the strip and markers stay. ThemeOptionList.row_width() is the content width minus the vertical scrollbar (always reserved: the catalog always overflows and a scrollbar appearing sends no Resize) and option padding; on resize the picker refits prompts in place (replace_option_prompt_at_index) rather than rebuilding the list, so a resize never moves the highlight or clears an export result. Test: test_long_theme_names_stay_on_one_row_at_80x24 (both Textual themes) plus test_row_truncates_only_the_name_to_fit. Files: settings_theme_picker.py, settings.md.

P3 review fix (I2): an active, launch-default theme that overrides a built-in rendered at 80x24 as '…  ▮▮▮▮▮▮▮  active · launch · overrides built-in' and still wrapped. _row now has a width budget: the name keeps min(len, 10) cells and the tail gives way first — 'overrides <origin>' -> 'overrides' -> dropped -> strip 7 -> 3 swatches -> 'launch' dropped; 'active' never drops. Tests: test_row_tail_gives_way_before_the_name[32/26] and test_active_launch_overriding_theme_stays_on_one_row_at_80x24 (both fail on the pre-fix _row).
<!-- SECTION:NOTES:END -->
