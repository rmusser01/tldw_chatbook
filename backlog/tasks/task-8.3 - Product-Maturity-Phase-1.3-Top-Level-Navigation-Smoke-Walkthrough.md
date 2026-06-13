---
id: TASK-8.3
title: 'Product Maturity Phase 1.3: Top-Level Navigation Smoke Walkthrough'
status: Done
assignee: []
created_date: '2026-05-05 17:29'
updated_date: '2026-05-05 17:45'
labels:
  - product-maturity
  - phase-1-qa-baseline
dependencies:
  - TASK-8.2
parent_task_id: TASK-8
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify that every top-level destination can be reached from a clean running app and presents usable orientation, recovery, or explicit blocker states instead of render-only shells.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh clean-run walkthrough evidence records all top-level navigation destinations and whether each is usable, blocked, or degraded.
- [x] #2 Every visible top-level destination either reaches its intended screen or records a P0/P1 finding under the product-maturity severity policy.
- [x] #3 Keyboard and focus notes cover command palette or navigation fallback for top-level destination access.
- [x] #4 Focused regression coverage protects the top-level navigation smoke evidence and tracker/task closeout seams.
- [x] #5 The Phase 1 tracker and QA index distinguish this top-level navigation gate from the remaining visual, empty-state, and core-loop gates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused failing Phase 1.3 navigation smoke contract test that expects the new task, tracker, and evidence links.
2. Extend the Textual clean-run pilot walkthrough to enumerate visible top-level navigation buttons and verify each route reaches an intended screen or explicit blocker/orientation state.
3. Create Phase 1.3 QA evidence using the Phase 1 template, including keyboard/focus notes and destination status table.
4. Update the Phase 1 README and product-maturity tracker with Phase 1.3 status while keeping Phase 1 open for remaining gates.
5. Run focused UI verification, mark TASK-8.3 acceptance criteria complete, and document implementation notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified top-level destination reachability from a clean first-run Textual app pilot. Added a focused Phase 1.3 navigation smoke regression that enumerates the canonical master-shell destination order, activates every top-level nav button, waits for the expected screen/current-tab route, and locks the evidence/tracker/task closeout seams. Added Phase 1.3 QA evidence and tracker/README updates while keeping Phase 1 open for keyboard/focus, visual broken-state, empty/error/setup-state, and core-loop gates.
<!-- SECTION:NOTES:END -->
