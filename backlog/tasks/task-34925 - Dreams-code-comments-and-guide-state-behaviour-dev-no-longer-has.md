---
id: TASK-34925
title: Dreams code comments and guide state behaviour dev no longer has
status: To Do
created_date: 2026-10-12 23:18
labels:
- dreams
- docs
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After the URL-provenance rework merged with PR #2921, several statements contradict the code. A track-service docstring says a restored unmarked source is fetched as a trusted origin (it is not) and gives a boot-path cost that is not the real reason. The startup-reconcile docstring names two of the three stores it asks and gives no reason for checking once instead of holding a scope. The user guide says a user's own sources are fetched exactly as before, private address or not, which is false since every source a user made before the rework reads as unknown, and its restore paragraph does not say what a restore brings back. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every statement listed in the review report about egress provenance, restores and the startup reconcile matches dev's behaviour
- [ ] #2 The guide says how a user's own private sources are treated after the provenance rework
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
