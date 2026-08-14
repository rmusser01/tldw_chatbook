---
id: TASK-16001
title: Make Context arrow a full button label
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-14 02:19'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the collapsed Console Context affordance read as one obvious, full-width clickable control by combining its label and ASCII arrow without widening the rail or changing vertical, tooltip, or responsive behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The horizontal collapsed Context button displays exactly Context---> on one composited row within its existing eleven-cell content width.
- [ ] #2 The entire Context---> surface is one button, including the final arrow cell, and opens the Context rail.
- [ ] #3 The Open Context rail tooltip and vertical Context presentation remain unchanged.
- [ ] #4 Inspector rail labels, rail widths, IDs/classes, open rails, and responsive behavior remain unchanged.
- [ ] #5 Focused Context rail, interaction, compact-access, visual, lint, formatting, and integrity checks pass.
<!-- AC:END -->

## Design

<!-- SECTION:DESIGN:BEGIN -->
Approved design: `Docs/superpowers/specs/2026-08-13-task-16001-context-arrow-button-design.md`.
<!-- SECTION:DESIGN:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED pure, mounted, compact arrow-end interaction, saved-style, settings, core-loop, and three-viewport compositor regressions for the exact `Context--->` button while preserving vertical, noncanonical, Inspector, width, tooltip, and shared-handle contracts.
2. Translate only the canonical horizontal Context label in `ConsoleRailHandle._display_label()` and clear Textual's default line padding inline only on the existing horizontal left Button so the eleven-cell literal fits the unchanged eleven-cell content region.
3. Mutation-check the last-cell click, run only directly modified/related tests plus focused Ruff, formatter, duplicate-ID, and diff checks, then self-review and close the task with fresh evidence.

ADR required: no.

ADR path: N/A.

Reason: this is a reversible presentation refinement inside the existing Console-specific rail display seam and changes no architecture, persistence, dependency, security policy, public interface, or ownership boundary.

Detailed plan: `Docs/superpowers/plans/2026-08-13-task-16001-context-arrow-button.md`.
<!-- SECTION:PLAN:END -->
