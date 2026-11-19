---
id: TASK-35381
title: Apply client Canvas sync through a durable pending inbox
status: To Do
created_date: 2026-10-18 05:56
dependencies:
- TASK-35380
labels:
- canvas
- sync
- client
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Receive inert Canvas histories through the existing domain owner without losing dependency-delayed records or silently resolving conflicts. Preserve local branch and browser selection while applying the approved replica lifecycle contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Incoming envelopes enter durable custody before pull-cursor advancement and apply through CanvasService, not direct transport SQL; parent/origin dependencies across pages and crashes remain retryable without foreign-owner admission.
- [ ] #2 Immutable replay, competing offline siblings and quota overage preserve exact candidates; conflicts are durable and source-free in summaries, and explicit resolution appends a new continuation rather than overwriting or pruning history.
- [ ] #3 Delete/restore epochs, parent-conversation lifecycle and retired-owner fences follow ADR-239 in both delivery orders; unknown profiles stay inert and newer payloads remain safely pending without downgrade or echo.
- [ ] #4 Tests verify crash/apply/receipt boundaries, no-echo, deterministic active-branch projection, untouched local selection and zero renderer/browser/compile or capability creation during synchronization.
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
