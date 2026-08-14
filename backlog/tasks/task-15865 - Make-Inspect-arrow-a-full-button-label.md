---
id: TASK-15865
title: Make Inspect arrow a full button label
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-13 17:12'
updated_date: '2026-08-13 17:12'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the collapsed Console Inspector affordance read as one obvious, full-width clickable control by combining its short label and repeated arrow into `Inspect-->`, without widening the rail or changing its badge and compact-layout behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The horizontal collapsed Inspector button displays exactly `Inspect-->` on one composited row within its existing nine-cell content width.
- [ ] #2 The entire `Inspect-->` surface is one button that opens the Inspector rail and retains the `Open Inspector rail` tooltip.
- [ ] #3 The optional approval badge remains a separate row beneath the button with its existing abbreviation and containment behavior.
- [ ] #4 The vertical `Inspector` presentation, left Context handle, rail widths, open rail, and responsive behavior remain unchanged.
- [ ] #5 Focused Console rail, interaction, compact-access, and visual regressions pass.
<!-- AC:END -->

## Design

<!-- SECTION:DESIGN:BEGIN -->
Approved design: `Docs/superpowers/specs/2026-08-13-task-15865-inspector-arrow-button-design.md`.
<!-- SECTION:DESIGN:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: N/A

Reason: this is a reversible display-copy refinement inside the existing Console rail presentation seam.

1. Add RED component, mounted, arrow-end interaction, and six-state compositor expectations for the exact `Inspect-->` button.
2. Change only the canonical horizontal Inspector display literal in `ConsoleRailHandle._display_label()`.
3. Run directly related rail, compact-access, CSS-integrity, visual, static, and duplicate-task-ID checks; do not run the full repository suite per user instruction.
4. Self-review, record fresh evidence, complete AC #1-5, and mark TASK-15865 Done only if every scoped gate is green.

Detailed plan: `Docs/superpowers/plans/2026-08-13-task-15865-inspector-arrow-button.md`.
<!-- SECTION:PLAN:END -->
