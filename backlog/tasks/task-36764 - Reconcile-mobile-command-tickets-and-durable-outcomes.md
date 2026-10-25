---
id: TASK-36764
title: Reconcile mobile command tickets and durable outcomes
status: To Do
created_date: 2026-10-25 17:39
dependencies:
- TASK-36760
- TASK-36762
- TASK-36763
labels:
- console
- web-server
- mobile
- adr-244
priority: high
documentation:
- Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md
- Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md
- backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent repeated or uncertain phone submissions from causing duplicate Console actions while keeping native cancellation available and outcomes honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Authenticated server-issued tickets bind action, target and owner epochs to an immutable five-minute monotonic submission expiry; tampered, expired and prior-epoch tickets cannot issue effects.
- [ ] #2 An exact command is admitted at most once, changed payload reuse is refused, and preparing, accepted, queued, applied, refused and unknown outcomes reflect native custody rather than transport acknowledgement.
- [ ] #3 Valid-ticket evidence and pending or unknown outcomes cannot be evicted into replay; capacity refuses new non-idempotent effects while exact-target native Stop remains available under receipt pressure.
- [ ] #4 Receipts needed after owner loss are durable and body-free, completed evidence prunes only after expiry, and executed schema migrations plus exact recovery catalogs preserve them.
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
