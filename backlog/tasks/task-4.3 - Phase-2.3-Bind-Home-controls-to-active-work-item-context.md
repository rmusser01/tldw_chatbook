---
id: TASK-4.3
title: 'Phase 2.3: Bind Home controls to active-work item context'
status: Done
assignee: []
created_date: '2026-05-03 18:04'
updated_date: '2026-05-03 18:08'
labels:
  - unified-shell
  - phase-2
  - home
  - adapter
  - ux
dependencies: []
parent_task_id: TASK-4
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Home active-work controls operate on explicit visible item context instead of anonymous aggregate counts, so future service adapters can approve, pause, resume, retry, open details, or launch Console for a specific work item.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Home dashboard state can carry active-work item identity, title, source, status, detail route, and Console availability.
- [x] #2 Home active-work summary renders visible item context before controls are used.
- [x] #3 Home controls carry target item ids and pass them through app hooks into the adapter.
- [x] #4 Existing count-only unavailable states remain supported for backward compatibility.
- [x] #5 Focused pure state, adapter, and mounted Home tests cover item-scoped controls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing pure dashboard, adapter, and mounted Home tests for item-scoped active-work controls.
2. Extend Home dashboard state with an active-work item model and target-aware controls while preserving count-only fallback behavior.
3. Pass target item ids through HomeScreen app hooks and the active-work adapter contract.
4. Add Phase 2 QA evidence plus roadmap and task updates.
5. Run focused Home/Phase 2 tests and diff hygiene before committing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added HomeActiveWorkItem context for Home dashboard state, derived visible active-work rows and target-aware controls from item status, preserved the existing count-only fallback path, and threaded target_id through HomeScreen, TldwCli Home hooks, and HomeActiveWorkAdapter.handle_control. Added focused pure state, adapter, mounted Home, and Phase 2 evidence-link coverage plus QA documentation.
<!-- SECTION:NOTES:END -->
