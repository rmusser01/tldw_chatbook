---
id: TASK-139
title: Implement Chatbook Workbench UI foundation and Console reference
status: Done
assignee: []
created_date: '2026-06-29 06:34'
updated_date: '2026-06-29 18:36'
labels:
  - ui
  - textual
  - architecture
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the first implementation slice for the Posting-inspired Chatbook workbench redesign so the app gains a measured responsiveness baseline, shared Textual UI primitives, discoverable focus/help conventions, and a Console reference implementation before broader route migrations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ADR records the Chatbook Workbench UI System decision
- [x] #2 Responsiveness instrumentation captures event-loop, worker, timer, and mount churn baseline
- [x] #3 Shared workbench primitives support stable composition, visible state, and normal plus compact density
- [x] #4 Console replacement reaches feature parity before legacy retirement
- [x] #5 Core Console workflow is completable without command palette
- [x] #6 Route inventory coverage is tracked for future migration owners
- [x] #7 Snapshot, interaction, and soak verification gates are defined and runnable
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes

ADR path: `backlog/decisions/011-chatbook-workbench-ui-system.md`

Reason: shared UI architecture, route ownership, focus/help conventions, and responsiveness gates are long-lived application structure.

Plan: `Docs/superpowers/plans/2026-06-29-chatbook-workbench-ui-foundation-console-plan.md`

1. Add route inventory and migration-owner coverage for all registered routes and aliases.
2. Add UI responsiveness instrumentation for heartbeat lag, workers, timers, and mount churn.
3. Build shared Workbench state, widget, focus, help, and density primitives.
4. Refactor Console as the reference implementation while keeping its left rail, transcript, inspector, and composer framing.
5. Verify discoverability, Console parity, responsiveness gates, navigation, and task documentation before marking Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the first Workbench UI redesign slice: ADR-linked route ownership, responsiveness instrumentation, shared Workbench state/widgets, focus/help conventions, and Console as the reference implementation. Console now exposes core workflow controls visibly in the Workbench frame while preserving the existing left context rail, transcript surface, inspector rail, and composer layout framing.

Verification added route inventory, responsiveness, visual snapshot, Console contract, parity matrix, navigation, CSS build, diff hygiene, and route-switch soak gates. A stale missing-model recovery test was aligned with the visible Workbench recovery action, and `TldwCli.query_one()` now preserves the original query miss when no active screen exists instead of surfacing a startup `ScreenStackError`.

Later destination migrations remain intentionally scoped to future tasks using the route-owner coverage introduced here. Final evidence is recorded in `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console.md`.
<!-- SECTION:NOTES:END -->
