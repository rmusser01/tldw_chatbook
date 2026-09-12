---
id: TASK-15666
title: busy_fleet_session_count prunes the fleet as a side effect of a read
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-11 21:30'
updated_date: '2026-09-12 07:28'
labels:
  - console
  - agents
  - threading
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The navigation confirmation uses busy_fleet_session_count to inspect live work. That read must leave coordinator handles and retained service owners unchanged, while normal rail, cancellation, lifecycle, and between-turn cleanup still release settled state. Current revalidation found terminal handles already pruned between turns; the remaining mutation was retained-owner cleanup inside fleet_snapshot.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The busy count is derived without mutating coordinator state
- [x] #2 Pruning still happens where it did before (between turns), on the same schedule
- [x] #3 A test asserts that taking the busy count leaves the handle set unchanged
- [x] #4 Taking the busy count does not remove retained survivor owners; existing lifecycle and rail cleanup still release settled owners.
- [ ] #5 Repeated headless turns release settled retained service owners at the next turn boundary without a rail read; live survivors remain retained and stoppable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/129-fleet-mailbox-and-wake-reliability.md (existing contract)
Reason: restore the documented read-only fleet snapshot while preserving ownership, coordinator-handle pruning timing, and storage; settled-owner cleanup gains an explicit headless turn boundary.
1. Revalidate the controller-to-bridge read path on current dev. Terminal coordinator pruning is already between turns, but fleet_snapshot still prunes retained services.
2. Extend the real bridge/controller survivor regression to assert coordinator handles and retained owners are unchanged by busy counts before and after child settlement.
3. Remove cleanup from fleet_snapshot and keep existing live_snapshot/cancellation/lifecycle cleanup; update comments and the affected cleanup regression.
4. Final revalidation amendment: prune settled retained services at the start of _conversation_fleet_coordinator, before the fleet-disabled early return and outside admission locks. Prove repeated headless turns reclaim prior settled owners while retaining live owners; keep terminal-handle pruning unchanged.
5. Run targeted survivor, count, and between-turn pruning tests plus changed-line lint/format checks and independent review; record evidence before Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Made ConsoleAgentBridge.fleet_snapshot observational by removing its retained-owner cleanup call; service lookup and live survivor filtering remain unchanged. The real controller/bridge regression verifies live and terminal busy-count reads preserve both retained owners and coordinator handles. Existing live_snapshot cleanup still releases settled owners, and the next turn still prunes terminal handles.
ADR required: no new ADR; existing backlog/decisions/129-fleet-mailbox-and-wake-reliability.md applies. Updated bridge/controller lifecycle comments and the affected survivor cleanup assertion.
Verified commit 726409de10 with 10 targeted tests; the strengthened regression failed on the original retained-owner mutation and passed after the fix. Independent task review approved spec and quality. Zero changed-line Ruff findings and git diff --check passed. Whole-file Ruff/format debt and environment dependency/temp-cleanup warnings remain; no guard or lint threshold was raised.

Final lifecycle revalidation exposed an incomplete cleanup transfer: two headless turns retained two settled owners because coordinator pruning removed handles only. Reopened before integration to move settled-owner pruning to the existing between-turn lifecycle boundary. The fleet snapshot remains observational.
<!-- SECTION:NOTES:END -->
