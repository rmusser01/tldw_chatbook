---
id: TASK-16001
title: Make Console rail controls directional full buttons
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-14 02:19'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make both Console rails communicate expansion and collapse direction correctly while replacing each open rail's tiny arrow target with one full-width, clearly clickable header control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Horizontal collapsed handles display exactly `Context->` and `<-Inspect` on one row, with arrows pointing inward and the full labels opening their respective rails.
- [ ] #2 Open Context and Inspector headers are single full-width Buttons labeled exactly `<---------|Context` and `Inspect|--------->`, with arrows pointing outward and every painted cell collapsing its rail.
- [ ] #3 Existing open/collapse Button IDs, tooltips, keyboard focus behavior, persistence, vertical labels, and noncanonical-label behavior remain unchanged.
- [ ] #4 Rail widths, bodies, Inspector badges, shared/Lab/Personas rails, responsive rules, transcript access, and terminal frames remain unchanged.
- [ ] #5 Focused component, mounted interaction, compact-access, compositor, focus-tour, lint, formatting, CSS integrity, duplicate-ID, and diff checks pass without running the full repository suite.
<!-- AC:END -->

## Design

<!-- SECTION:DESIGN:BEGIN -->
Approved design: `Docs/superpowers/specs/2026-08-13-task-16001-console-rail-directional-buttons-design.md`.
<!-- SECTION:DESIGN:END -->

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
