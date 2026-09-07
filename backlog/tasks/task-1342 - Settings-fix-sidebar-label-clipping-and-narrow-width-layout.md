---
id: TASK-1342
title: Settings fix sidebar label clipping and narrow-width layout
status: Done
assignee: []
created_date: '2026-08-04 23:47'
updated_date: '2026-08-05 02:44'
labels:
  - settings
  - ux
  - layout
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
.settings-category-button CSS forces height:1/max-height:1 (_agentic_terminal.tcss:3424): 'Providers & Models' renders as 'Providers &' and 'Privacy & Security' as 'Privacy &' at <=120 cols, and the dirty * marker is clipped off edited categories. At 80x24 the category list disappears entirely and form controls are focusable but rendered off-screen (keyboard users operate blind).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All 19 category labels render fully at 120x35 and 100x30 including the dirty marker
- [x] #2 At 80x24 the screen offers a working narrow layout (visible categories or explicit filter-first mode)
- [x] #3 Inspector text does not wrap mid-word at supported widths
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read UAT evidence snapshots and settings_screen.py sidebar/inspector build code
2. Find narrow-width precedent in other screens (grep CSS for width guards)
3. Write failing tests: full label rendering incl. dirty marker at 120x35/100x30, working narrow mode at 80x24, no mid-word inspector wraps
4. Implement CSS/structure fix following existing conventions
5. Run new tests + category sweep + configuration hub + footer hints + save/commit models tests
6. Self-review, update task notes, mark Done
ADR required: no
ADR path: N/A
Reason: layout/CSS fix only, no architectural decision
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- **Root cause (label clipping):** Textual 8 `Button` defaults to `line-pad: 1`, reserving one cell on each side of the label, so a 22-cell label needed 24 cells of content width the 3fr sidebar didn't have; the label wrapped and `height: 1; max-height: 1` clipped the second line ("Providers &" / lost `*`). `line-pad: 0` is not expressible in Textual CSS (its `_process_integer` rejects 0), so the fix is width guarantees instead: `#settings-category-pane { min-width: 32 }` (label 22 + line-pad 2 + border 2 + padding 2 + scrollbar 1 + button padding 2, plus margin) and `#settings-impact-pane { min-width: 21 }` (longest inspector word "Read-only/WIP;" = 14 + chrome).
- **Narrow mode (AC2):** followed the existing `personas-workbench-compact` precedent (`personas_screen.py` `_sync_responsive_workbench`, threshold 90). New `SETTINGS_COMPACT_WORKBENCH_MAX_WIDTH = 90`; `on_resize` + `on_mount` toggle `settings-workbench-compact` / `settings-workbench-compact-pane` classes, and `compose_content` applies them at compose time so the `recompose=True` reactives can't drop them. Compact CSS: category pane fixed 32-wide sidebar (padding 0 1), detail pane takes the rest (1fr, padding 0 1), inspector pane + its divider `display: none`, and the search help/status rows hide so ~6 category rows stay visible at 24-row heights. Categories remain visible (not filter-first); the filter input keeps its "Filter settings (/)" placeholder and Enter-to-open.
- **Mid-word wraps (AC3):** fixed by the inspector `min-width: 21`; words now wrap on whitespace at 100x30 and 120x35. At 80x24 the inspector is hidden in compact mode.
- **TDD:** new `Tests/UI/test_settings_narrow_layout.py` (9 tests) loads the REAL stylesheet via a `DestinationHarness` subclass with `CSS_PATH = tldw_cli_modular.tcss` -- the shared harness runs without app CSS and cannot see these defects. Confirmed 8/9 failing before the fix, all green after. Inspector test scrolls the pane by content height to cover all rows.
- **Files changed:** `tldw_chatbook/css/components/_agentic_terminal.tcss` (min-widths, compact rules, comments), `tldw_chatbook/css/tldw_cli_modular.tcss` (regenerated via `build_css.py`), `tldw_chatbook/UI/Screens/settings_screen.py` (compact constant + sync + compose classes), `Tests/UI/test_settings_narrow_layout.py` (new).
- **Verified:** 9/9 new tests; Tests/UI/test_settings_category_sweep.py + test_settings_configuration_hub.py + test_settings_footer_hints.py + test_settings_save_commit_models.py = 260 passed; remaining 6 settings UI files = 53 passed; text snapshots at 120x35/100x30/80x24 eyeballed (full "Providers & Models *", visible 80x24 sidebar, clean inspector wraps). Ruff: 2 pre-existing findings in settings_screen.py (`save_setting_to_cli_config` F401/F811) untouched by this change.
- **ADR:** not required (layout/CSS fix only); linked precedent: personas compact workbench.
<!-- SECTION:NOTES:END -->
