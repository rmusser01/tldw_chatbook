---
id: TASK-36760
title: Keep served control and projection draining independent of phone clients
status: To Do
created_date: 2026-10-25 17:39
dependencies:
- TASK-36729
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
Make the authenticated process control channel available independently of Canvas and mobile visits, and prevent projection congestion from becoming execution-owner loss.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every admitted child has authenticated runtime control and lifecycle-owned event draining before publication, even with Canvas disabled, the mobile surface disabled or no phone subscriber.
- [ ] #2 Projection overflow uses bounded coalescing or reset state without closing the host control channel, blocking agent execution, dropping command outcomes or resolving native decisions.
- [ ] #3 Snapshots and events expose a detectable sequence gap and recover current state; malformed protocols, revoked child credentials and genuine control loss remain distinct failures.
- [ ] #4 Existing Canvas authentication, feature revocation and browser capability isolation remain enforced under the revised control protocol.
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
