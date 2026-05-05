---
id: TASK-8.5
title: 'Product Maturity Phase 1.5: Visual Broken-State Audit'
status: Done
assignee: []
created_date: '2026-05-05 18:54'
updated_date: '2026-05-05 18:59'
labels:
  - product-maturity
  - phase-1-qa-baseline
dependencies:
  - TASK-8.4
parent_task_id: TASK-8
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the clean-run shell visually survives supported terminal sizes without blank, traceback, collapsed, or unexplained broken states across top-level destinations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh clean-run evidence records visual/chrome behavior across compact laptop and large terminal sizes
- [x] #2 Top-level destinations render non-empty content with navigation chrome and fallback affordance intact
- [x] #3 Any P0/P1 visual broken-state findings are fixed or explicitly accepted under the product-maturity severity policy
- [x] #4 Screenshot or SVG evidence is captured when it materially clarifies layout or visual defects
- [x] #5 Focused regression coverage protects the visual audit evidence and tracker/task closeout seams
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused failing Phase 1.5 visual-audit contract test for SVG export, destination content, and task/tracker links.
2. Run the contract to confirm the missing Phase 1.5 evidence and tracking fail first.
3. Exercise the running Textual app across compact, laptop, and large terminal sizes for every top-level destination.
4. Create Phase 1.5 QA evidence and update the Phase 1 README and product-maturity tracker while keeping Phase 1 open for remaining gates.
5. Mark TASK-8.5 acceptance criteria complete after focused verification and document implementation notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified clean-run visual/chrome integrity across compact (100x32), laptop (140x40), and large (180x50) terminal sizes for every top-level destination. Added a focused Phase 1.5 regression that exports in-memory SVG screenshots, waits for async chrome stabilization, checks non-empty destination content, validates shared navigation/fallback affordance, and guards against traceback/raw-object broken states. Added Phase 1.5 QA evidence and tracker/README updates while keeping Phase 1 open for empty/error/setup-state and core-loop gates.
<!-- SECTION:NOTES:END -->
