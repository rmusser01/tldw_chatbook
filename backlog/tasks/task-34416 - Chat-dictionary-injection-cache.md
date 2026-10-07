---
id: TASK-34416
title: Chat-dictionary injection cache
status: To Do
created_date: 2026-10-07 02:40
dependencies:
- TASK-34415
updated_date: 2026-10-07 02:43
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 2 / F4: dictionary entries are re-loaded from DB re-instantiated and regex-recompiled on every send and twice per turn
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Store generation counters added,Second send with unchanged dictionaries does no DB loads or regex compiles,Replacement output byte-identical on golden fixture,Console-seam double collection noted for coordination
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 4 (T4)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
