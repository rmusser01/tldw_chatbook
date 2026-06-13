---
id: TASK-12.3
title: 'Phase 5.3: Server events and notifications live feed'
status: Done
assignee: []
created_date: '2026-05-16 00:00'
updated_date: '2026-05-16 15:44'
labels:
  - product-maturity
  - phase-5-server-parity
dependencies: []
parent_task_id: TASK-12
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Surface server event and notification presentation state in user workflows without making local notification state authoritative for server-owned resources.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 QA walkthrough verifies local notifications and server-owned event presentation are visibly distinct.
- [x] #2 Focused regression evidence covers replay gap, reconnect/requery, and server unavailable event states.
- [x] #3 Repo-tracked QA evidence exists under Docs/superpowers/qa/product-maturity/phase-5/.
- [x] #4 P0/P1 findings are fixed or explicitly accepted according to the spec severity policy.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused regressions for Home server-event presentation states: available feed, replay gap requiring server requery, missing active server scope, and unavailable server event backend. 2. Extend Home dashboard state with server-event count/state/recovery copy while keeping local notification count separate. 3. Wire LocalNotificationHomeActiveWorkAdapter to existing NotificationsScopeService observed-feed projection without adding transport or local authority. 4. Wire app service initialization so Home receives the observed server-event presentation service. 5. Add Phase 5.3 QA evidence and tracker/task updates. 6. Run focused Notifications/Home/Phase 5 verification and diff hygiene before PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added server-event presentation fields to Home dashboard input/state while keeping local notification count separate.
- Wired the Home adapter to the existing Notifications scope observed-feed projection with `mark_presented=False`, so Home rendering does not mutate server-owned presentation/read state.
- Added focused Home adapter regressions for mixed local/server notification state, replay-gap requery recovery, reconnect-required scope failure, and unavailable server event backend.
- Added Phase 5.3 QA evidence and updated the Phase 5 tracker/readme/test contract.
<!-- SECTION:NOTES:END -->
