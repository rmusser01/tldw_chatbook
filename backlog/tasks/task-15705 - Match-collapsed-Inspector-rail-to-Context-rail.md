---
id: TASK-15705
title: Match collapsed Inspector rail to Context rail
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-13 06:07'
updated_date: '2026-08-13 13:42'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the collapsed Console Inspector rail visually match the established Context rail treatment so its label is centered vertically and its background fills the full rail column.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The collapsed Inspector handle fills the Console workspace column vertically.
- [ ] #2 The collapsed Inspector column uses the same filled panel and border treatment as the collapsed Context column.
- [ ] #3 The Inspector label is centered vertically while its optional badge remains visible and legible.
- [ ] #4 Existing Inspector width, tooltip, badge abbreviation, open behavior, and compact-width access remain unchanged.
- [ ] #5 Component TCSS and the generated stylesheet remain in sync.
- [ ] #6 Focused Console rail and stylesheet integrity tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/017-console-left-rail-usability.md
Related ADR: backlog/decisions/043-console-rail-compact-collapse-yields-to-explicit-toggle.md
Reason: Existing ADR-017 governs the rail visual language and ADR-043 governs compact-width access preserved by AC #4; no new architectural decision is introduced.

1. Add solid-framed, production-stylesheet regressions for Context/Inspector height parity, deterministic unbadged/badged geometry, shared-right-handle isolation, and source/bundle CSS parity.
2. Add a Console-only Inspector handle class and Python button geometry override; switch its Console frame from quiet to solid.
3. Update component TCSS, regenerate the production CSS bundle, and run focused rail/CSS/static checks.
4. Run a real-Console six-state SVG/geometry sweep at 100x30, 140x42, and 160x45 with zero/three approvals.
5. Run full-suite Ruff/check/format gates; verify the task diff; close only if every Definition of Done gate is green.
<!-- SECTION:PLAN:END -->
