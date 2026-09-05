---
id: TASK-31642
title: Restore Console screen decomposition ratchets
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 16:34'
updated_date: '2026-09-05 16:36'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move cohesive Console responsibilities into the established controller and region boundaries so the current screen satisfies its existing size contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All Console Architecture size checks pass without raising existing ceilings.
- [ ] #2 Moved behaviors retain late-bound dependencies and existing Textual ownership.
- [ ] #3 Focused Console tests and static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/033-application-session-state-ownership.md
Reason: Mechanical ownership extraction implements DESIGN.md section 7 and the approved screen decomposition design, preserving existing runtime and persistence boundaries.
1. Move settings durability into a named controller, preserve app lifetime admission and test the settings/default flows.
2. Move settings navigation, provider selection, row menus, and cohesive projection clusters sequentially with explicit late-bound dependencies.
3. Remove obsolete screen forwarding methods after updating their production callers; preserve Textual event and lifecycle edges.
4. Run focused Console behavior and Architecture checks, inspect final counts, and record exact evidence.
<!-- SECTION:PLAN:END -->
