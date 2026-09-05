---
id: TASK-31567
title: Refresh Watchlists regression contracts
status: Done
assignee: []
created_date: '2026-09-05 03:29'
updated_date: '2026-09-05 04:08'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Align Watchlists UI regression coverage with the current safe error copy, navigation context, status paging, and source-form field contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The four reproduced Watchlists failures pass.
- [x] #2 The affected Watchlists modules pass in full.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce and inspect the four Watchlists failures against current production behavior. 2. Refresh assertions or harness interactions only where shipped contracts are intentional; fix production only if a behavior defect is confirmed. 3. Run focused cases and all affected Watchlists modules. ADR required: no. ADR path: N/A. Reason: This is regression coverage maintenance for existing UI behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Refreshed Watchlists regression coverage for current navigation ownership, safe error copy, reader paging, run retry eligibility, and backend-specific source forms. Replaced obsolete controller and private-loader test seams with the shipped paged Reader API and snapshot refresh path. Fixed a production regression where a same-context manual refresh could discard the currently open Reader item. Stabilized the cold-loader test by waiting for the preceding surface refresh to drain before rewinding its state. Verified all 152 tests in the three affected Watchlists modules; the run passed with two dependency warnings and one existing file-descriptor-growth warning. ADR required: no. ADR path: N/A. Reason: the change restores existing Reader refresh behavior and updates regression contracts without changing an architectural boundary.
<!-- SECTION:NOTES:END -->
