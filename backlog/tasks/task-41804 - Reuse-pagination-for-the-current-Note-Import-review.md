---
id: TASK-41804
title: Reuse pagination for the current Note Import review
status: In Progress
created_date: 2026-11-18 20:44
assignee:
- '@codex'
labels:
- performance
- notes
updated_date: 2026-11-18 21:21
documentation:
- Docs/QA/console-library-responsiveness/README.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Page navigation in Note Import review should spend work on the requested page, with all existing group counts and review actions preserved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Page navigation reuses layout for an unchanged immutable plan and page size.
- [x] #2 Replacement plans and collision or item changes retain the clamped page, actions, and complete group and run totals.
- [x] #3 Paired measurements cover the page model and mounted Import Review paging in the Terminal harness.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin reuse on page moves and rebuilding on plan or size changes.
2. Attach immutable page boundaries and counts to the current review page. Select the original page in one pass for replacement plans.
3. Verify collapsed runs, item decisions and collisions plus paired model and mounted Terminal navigation.

ADR required: no
ADR path: N/A
Reason: ADR-059 and the existing immutable preview and approval owner still govern import.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The current immutable plan and page size own the derived page boundaries and counts. Navigation reuses the page layout, and a replacement plan computes the clamped selection in one pass. State and controller tests passed for complete totals, collisions and item decisions. In three mounted pairs, page calculation fell from 3.770 to 0.031 ms CPU, and median page paint rose from 211.8 to 229.5 ms. The same ordered page rendered in all six runs. ADR-059 still governs approval and execution. See the QA record. Hosted verification remains open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
