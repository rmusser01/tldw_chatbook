---
id: TASK-16001
title: Make Console rail controls directional full buttons
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 02:19'
updated_date: '2026-08-14 05:01'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make both Console rails communicate expansion and collapse direction correctly while replacing each open rail's tiny arrow target with one full-width, clearly clickable header control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Horizontal collapsed handles display exactly `Context->` and `<-Inspect` on one row, with arrows pointing inward and the full labels opening their respective rails.
- [x] #2 Open Context and Inspector headers are single full-width Buttons labeled exactly `<---------|Context` and `Inspect|--------->`, with arrows pointing outward and every painted cell collapsing its rail.
- [x] #3 Existing open/collapse Button IDs, tooltips, keyboard focus behavior, persistence, vertical labels, and noncanonical-label behavior remain unchanged.
- [x] #4 Rail widths, bodies, Inspector badges, shared/Lab/Personas rails, responsive rules, transcript access, and terminal frames remain unchanged.
- [x] #5 Focused component, mounted interaction, compact-access, compositor, focus-tour, lint, formatting, CSS integrity, duplicate-ID, and diff checks pass without running the full repository suite.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Replace the superseded collapsed-Context RED expectations with exact inward `Context->` / `<-Inspect` contracts, then add real mounted coverage proving each open header is one full-width Button and title-side clicks collapse the correct rail.
2. Translate only the canonical horizontal collapsed labels in `ConsoleRailHandle`; replace each Console rail's separate title/tiny glyph pair with one existing-ID, one-row, full-width Button using the approved outward ASCII label and instance-local alignment.
3. Mutation-check both non-arrow title clicks, run only directly modified/related tests plus focused Ruff, format, Impeccable, CSS integrity, duplicate-ID, and diff checks, then self-review and close the task with fresh evidence.

ADR required: no.

ADR path: N/A.

Reason: this is a reversible Console-only presentation and hit-target correction using existing widgets, IDs, handlers, state, and layout boundaries; ADR-034's shared glyph ownership remains unchanged.

Detailed plan: `Docs/superpowers/plans/2026-08-13-task-16001-console-rail-directional-buttons.md`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented canonical inward collapsed labels and outward open-header labels for
both Console rails. Each open rail now exposes one existing-ID, full-width Button;
title-side click mutations proved the complete painted header is the hit target.
Vertical and noncanonical labels, badges, persistence, rail widths, responsive
behavior, shared consumers, and CSS remain unchanged.

The visual sweep also exposed a latent Textual midpoint-helper defect, corrected
to sample the upper middle row for even-height one-line controls. The related F6
tour was updated to tolerate the existing optional provider-recovery stop while
still requiring transcript confinement, status-chip reachability, Inspector focus,
and a ten-stop maximum.

Verification: 84 directly related regressions passed; 14 TASK-15783/TASK-16001
compositor cases passed; Ruff lint and format passed for all 14 related files;
Impeccable, CSS integrity, duplicate-task-ID, and diff checks passed. The full
repository suite was intentionally not run per user instruction.

ADR required: no. ADR path: N/A. This remains a reversible Console-only
presentation and hit-target change using existing state and interaction boundaries.
<!-- SECTION:NOTES:END -->

## Design

<!-- SECTION:DESIGN:BEGIN -->
Approved design: `Docs/superpowers/specs/2026-08-13-task-16001-console-rail-directional-buttons-design.md`.
<!-- SECTION:DESIGN:END -->
