---
id: TASK-31566
title: Refresh Settings category and Image Gen coverage
status: Done
assignee: []
created_date: '2026-09-05 02:49'
updated_date: '2026-09-05 03:10'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Align Settings regression coverage with the current writable Schedules category, three added categories, app-tier Appearance CSS, and scroll-owned Image Gen actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The seven reproduced Settings failures pass
- [x] #2 Both affected Settings modules pass in full
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update category count and ownership assertions to the current declared contracts. 2. Run the Appearance geometry test with the real Settings stylesheet. 3. Scroll Image Gen action buttons into view and disable splash in the real-App geometry test. 4. Run the focused cases and both affected modules. ADR required: no. ADR path: N/A. Reason: The changes update test harnesses and assertions to already-shipped Settings contracts without changing application behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Refreshed Settings coverage for the current 29-category inventory, writable Schedules ownership, and the third `priv` search match introduced by Personal Context. Appearance geometry and preview tests now load the app-tier stylesheet, while Image Gen Save/Revert interactions consistently scroll their action buttons into the Pilot viewport before clicking. The real-app Image Gen geometry test disables the splash through the normal setting seam. Verified all seven initially reproduced failures, the 16 additional failures exposed by the first full run, and both affected modules together (454 passed). ADR required: no; these are test-only updates to already-shipped contracts.
<!-- SECTION:NOTES:END -->
