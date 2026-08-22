---
id: TASK-20938
title: Repair Persona Buddy restart and frame sizing
status: In Progress
assignee: []
created_date: '2026-08-22 21:52'
labels: []
dependencies:
  - TASK-19055
references:
  - >-
    Docs/superpowers/specs/2026-08-22-persona-buddy-uat-repairs-design.md
  - >-
    backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the restart and portrait-cropping defects found during full-application Persona Buddy UAT so an explicitly configured Buddy restores faithfully and paints its complete resolved frame.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The effective `[persona_buddy]` table reaches the app-owned controller at startup, restoring the exact enabled state, local Persona selection, open/collapsed state, and saved geometry without a startup write.
- [ ] #2 Missing or malformed Buddy fields retain the existing strict independent safe defaults, and no unrelated configuration table or secret is added to the projected settings surface.
- [ ] #3 The first Workbench Buddy action after restart cannot replace valid saved geometry with the never-positioned sentinel merely because startup omitted persisted preferences.
- [ ] #4 Full-size Buddy resolution uses the visible `#persona-buddy-frame` content dimensions rather than the containing window dimensions; the complete prepared frame fits the painted slot without vertical cropping, and frame-slot size changes invalidate the exact resolution authority.
- [ ] #5 Collapsed, compact, hidden, detached, and stale views do not start an invalid zero-size resolution or repaint current views; existing animation, reduced-motion, fallback, cancellation, and navigation behavior remains unchanged.
- [ ] #6 Born-RED-to-GREEN focused tests, mutation proof for both root-cause guards, scoped static checks, and an isolated latest-dev full-app UAT without `NO_COLOR` prove color rendering, full-frame legibility, persistence across restart, and trusted operational state changes through a disposable local provider.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md`
Reason: ADR-074 already defines Buddy preference persistence, native Textual rendering, exact runtime authority, and verification boundaries; this task repairs two implementation defects without changing those decisions.

1. Review and approve the linked repair design, then write the executable TDD plan.
2. Add focused startup and frame-slot RED tests at the real configuration and Textual seams.
3. Apply the smallest root-cause fixes, run scoped regressions and mutations, and repeat isolated full-app UAT.
4. Record final evidence and deviations, complete the acceptance criteria, and close the task only when every scoped gate is green.
<!-- SECTION:PLAN:END -->
