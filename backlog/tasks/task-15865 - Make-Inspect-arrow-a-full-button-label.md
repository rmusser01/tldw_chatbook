---
id: TASK-15865
title: Make Inspect arrow a full button label
status: Done
assignee:
  - '@codex'
created_date: '2026-08-13 17:12'
updated_date: '2026-08-14 01:08'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the collapsed Console Inspector affordance read as one obvious, full-width clickable control by combining its short label and arrow into `Inspect->`, without widening the rail or changing its badge and compact-layout behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The horizontal collapsed Inspector button displays exactly `Inspect->` on one composited row within its existing nine-cell content width.
- [x] #2 The entire `Inspect->` surface is one button that opens the Inspector rail and retains the `Open Inspector rail` tooltip.
- [x] #3 The optional approval badge remains a separate row beneath the button with its existing abbreviation and containment behavior.
- [x] #4 The vertical `Inspector` presentation, left Context handle, rail widths, open rail, and responsive behavior remain unchanged.
- [x] #5 Focused Console rail, interaction, compact-access, and visual regressions pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: N/A

Reason: this is a reversible display-copy refinement inside the existing Console rail presentation seam.

1. Add RED component, mounted, arrow-end interaction, and six-state compositor expectations for the exact `Inspect->` button.
2. Change the canonical horizontal Inspector display literal in `ConsoleRailHandle._display_label()` and clear Textual's default `line_pad=1` inline on the existing horizontal right Button so the nine-cell label paints on one row. Runtime probing showed the original ten-cell `Inspect-->` direction could not fit the fixed nine-cell content width; the user selected `Inspect->` to preserve geometry.
3. Run directly related rail, compact-access, CSS-integrity, visual, static, and duplicate-task-ID checks; do not run the full repository suite per user instruction.
4. Self-review, record fresh evidence, complete AC #1-5, and mark TASK-15865 Done only if every scoped gate is green.

Detailed plan: `Docs/superpowers/plans/2026-08-13-task-15865-inspector-arrow-button.md`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the nine-cell Inspect arrow through the existing ConsoleRailHandle display seam using the fixed Inspect-> literal plus an inline line-pad reset on the existing horizontal right Button; no CSS-file, outer-width, child-widget, ID/class, or layout-structure changes were added. Modified console_rail_handle.py plus test_console_rail_handle.py, test_destination_rail.py, test_console_right_rail.py, test_console_shell_regions.py, test_settings_console_rail_labels.py, test_product_maturity_gate1_core_loop_screen_adaptation.py, and test_workbench_visual_snapshots.py. Directly related rail, interaction, compact-access, CSS-integrity, visual, Ruff, duplicate-ID, and diff checks passed. Per user instruction, no full repository suite was run. ADR required: no.
<!-- SECTION:NOTES:END -->

## Design

<!-- SECTION:DESIGN:BEGIN -->
Approved design: `Docs/superpowers/specs/2026-08-13-task-15865-inspector-arrow-button-design.md`.
<!-- SECTION:DESIGN:END -->
