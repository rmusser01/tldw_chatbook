---
id: TASK-3302
title: >-
  Ingest mode keyboard & focus: entry focus, Esc/i routes, F1/footer, focus visibility
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-07 19:30'
labels:
  - library
  - ingest
  - ux
  - accessibility
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Findings MI-03/04/05 (P1) of the 2026-08-07 Media Ingestion review. Three keyboard-first failures, all confirmed live: (1) entering Ingest mode leaves focus in the rail search box — typing a path runs a Library search (`_WORKBENCH_FOCUS_TARGETS` prefers a landing-only target); (2) there is no keyboard exit — Esc is bound only to skills-back, `i` is landing-scoped, the footer shows only generic hints, and F1 lists the screen's raw Skills BINDINGS (same BINDINGS-only-sourcing trap the Library P1 arc fixed for the footer — share the per-mode helper); (3) focus is structurally invisible on the path/title/author/keywords inputs and every compact button (`library-canvas-action`) — Tab produced byte-identical plain-text panes in two independent live walks, while panel fields show a heavy border. Color-only focus fails monochrome/low-contrast themes and the design system's three-signal dense-form convention.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Activating Ingest mode places focus on the path field; typing immediately edits the path (regression test)
- [ ] #2 Esc in Ingest returns to the Library browse/hub surface; `i` enters Ingest from any Library canvas
- [ ] #3 The Ingest footer and F1 help show the same per-mode ingest shortcuts (Enter start, Esc back, at minimum) from a shared source
- [ ] #4 Focused top-level ingest fields and compact action buttons show a structural (glyph-level, not color-only) focus indicator, asserted via plain-text render diff in a CSS-true harness
- [ ] #5 No focus/blur layout shift (dimensional stability preserved)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED tests: entry-focus assertion; Esc-in-ingest route; F1 content per mode; plain-text focus diff on a `library-canvas-action` button.
2. Add ingest-mode preferred focus target; per-mode escape handling; extend the footer/F1 shared shortcut helper with an ingest set.
3. Focus treatment: `outline: heavy $accent` (or heavy edge) on `.library-ingest-field:focus` and `.library-canvas-action:focus` in the app TCSS (rebuild bundle via build_css.py); keep DEFAULT_CSS parse-standalone.
<!-- SECTION:PLAN:END -->
