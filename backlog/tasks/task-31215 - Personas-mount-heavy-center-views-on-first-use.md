---
id: TASK-31215
title: Personas mount heavy center views on first use
status: In Progress
assignee: []
created_date: '2026-09-04 00:23'
updated_date: '2026-09-04 00:49'
labels:
  - roleplay
  - performance
dependencies: []
references:
  - >-
    backlog/tasks/task-2725 - Roleplay screen switch takes 2s where every other
    screen takes under 1s.md
  - backlog/tasks/task-31002 - Models-mount-only-the-active-provider-view.md
  - backlog/decisions/115-personas-demand-mounted-center-views.md
  - >-
    Docs/superpowers/specs/2026-09-03-personas-demand-mounted-center-views-design.md
  - Docs/superpowers/plans/2026-09-03-personas-demand-mounted-center-views.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the remaining Roleplay navigation stall by keeping the four heavy inactive center views out of the post-first-paint load path and mounting each only when its workflow is first activated, without losing restore, selection, editor, or Console-handoff state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening Personas reaches a usable initial Characters surface without mounting any inactive heavy center view.
- [ ] #2 First use mounts only the requested heavy view and revisiting reuses the same widget state.
- [ ] #3 Restore and deep-link intents replay after the required view is ready without applying stale state to another view.
- [ ] #4 Transient mount failures remain retryable and leaving the screen prevents stale callbacks from mutating detached UI.
- [ ] #5 Targeted Personas lifecycle and four-mode workflow tests pass, and a production-CSS responsiveness regression stays under the 250 ms event-loop-stall threshold.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red-first mounted tests proving real initial load leaves all four heavy roots absent and stable slots preserve document order.
2. Implement screen-owned first-use construction, mount, cache, hydration, retry, and lifecycle-generation behavior.
3. Add red-first workflow tests and route character/persona create/edit plus dictionary/lore select/create/edit/restore through the exact view admission boundary.
4. Add concurrency, transient-failure, teardown, and production-CSS 250 ms heartbeat regressions.
5. Run focused Personas and architecture verification, scoped static checks, self-review, and close the task with measured evidence.

ADR required: yes
ADR path: backlog/decisions/115-personas-demand-mounted-center-views.md
Reason: the change defines the long-lived lifecycle and restore/admission contract shared by the Personas screen and its four authoring widgets.
<!-- SECTION:PLAN:END -->
