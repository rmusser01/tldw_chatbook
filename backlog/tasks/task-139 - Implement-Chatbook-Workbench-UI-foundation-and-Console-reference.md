---
id: TASK-139
title: Implement Chatbook Workbench UI foundation and Console reference
status: In Progress
assignee: []
created_date: '2026-06-29 06:34'
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
- [ ] ADR records the Chatbook Workbench UI System decision
- [ ] Responsiveness instrumentation captures event-loop, worker, timer, and mount churn baseline
- [ ] Shared workbench primitives support stable composition, visible state, and normal plus compact density
- [ ] Console replacement reaches feature parity before legacy retirement
- [ ] Core Console workflow is completable without command palette
- [ ] Route inventory coverage is tracked for future migration owners
- [ ] Snapshot, interaction, and soak verification gates are defined and runnable
<!-- AC:END -->

## Implementation Plan

ADR required: yes

ADR path: `backlog/decisions/011-chatbook-workbench-ui-system.md`

Reason: shared UI architecture, route ownership, focus/help conventions, and responsiveness gates are long-lived application structure.

Plan: `Docs/superpowers/plans/2026-06-29-chatbook-workbench-ui-foundation-console-plan.md`

1. Add route inventory and migration-owner coverage for all registered routes and aliases.
2. Add UI responsiveness instrumentation for heartbeat lag, workers, timers, and mount churn.
3. Build shared Workbench state, widget, focus, help, and density primitives.
4. Refactor Console as the reference implementation while keeping its left rail, transcript, inspector, and composer framing.
5. Verify discoverability, Console parity, responsiveness gates, navigation, and task documentation before marking Done.
