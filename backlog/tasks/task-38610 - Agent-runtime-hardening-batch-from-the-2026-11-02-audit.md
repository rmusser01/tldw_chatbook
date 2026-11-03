---
id: TASK-38610
title: Agent runtime hardening batch from the 2026-11-02 audit
status: Done
created_date: 2026-11-02 14:30
labels:
- hardening
- agents
priority: low
updated_date: 2026-11-03 02:56
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Small hardening items in Agents and Tools: invoke_by_name can raise on non-finite floats despite its never-raise contract, engine _notify lacks the loop-closed guard its sibling has, the module-global write-lock map grows without bound, and the final hook drain stage waits without a deadline. See bug-sweep-2026-11-02-dev.md Agents and Tools findings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 invoke_by_name never raises on non-finite float arguments
- [x] #2 Engine notify gains the loop-closed guard
- [x] #3 Write-lock map is bounded or the growth is documented
- [x] #4 Final hook drain stage is bounded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
NaN-safe canonical args routed through the blocked channel; notify loop-closed guard with rollback; drain stage bounded at 8s; write-lock map capped with held locks never evicted plus identity recheck after acquire. Commit 26bc03e454.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
