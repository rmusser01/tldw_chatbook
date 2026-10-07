---
id: TASK-34418
title: Provider HTTP session reuse and payload deepcopy removal ADR-214
status: In Progress
created_date: 2026-10-07 02:41
dependencies:
- TASK-34417
updated_date: 2026-10-07 06:17
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 3 / F8b: full message-history payload is deepcopied per POST attempt and every provider call opens a fresh requests Session paying TLS handshake per turn
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ADR-214 written before code,Per-thread session registry with same-key reuse,No cross-thread session sharing,Payload passed by reference with mutation probe test,TLS handshake count evidence recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 6 (T6)
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
