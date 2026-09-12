---
id: TASK-15666
title: busy_fleet_session_count prunes the fleet as a side effect of a read
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-11 21:30'
updated_date: '2026-09-12 06:41'
labels:
  - console
  - agents
  - threading
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`busy_fleet_session_count` calls `fleet_snapshot`, which prunes terminal handles as a side effect, and it is called from the UI thread to build a navigation confirm. A read-shaped method on the UI thread should not mutate coordinator state that worker threads also write. The count itself was fixed in PR 3a-1 Task 6b (it previously reported "0 runs will be killed" and then killed one); this is about how it obtains the number.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The busy count is derived without mutating coordinator state
- [ ] #2 Pruning still happens where it did before (between turns), on the same schedule
- [ ] #3 A test asserts that taking the busy count leaves the handle set unchanged
- [ ] #4 Taking the busy count does not remove retained survivor owners; existing lifecycle and rail cleanup still release settled owners.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/129-fleet-mailbox-and-wake-reliability.md (existing contract)
Reason: restore the documented read-only fleet snapshot without changing ownership, pruning schedule, or storage.
1. Revalidate the controller-to-bridge read path on current dev. Terminal coordinator pruning is already between turns, but fleet_snapshot still prunes retained services.
2. Extend the real bridge/controller survivor regression to assert coordinator handles and retained owners are unchanged by busy counts before and after child settlement.
3. Remove cleanup from fleet_snapshot and keep existing live_snapshot/cancellation/lifecycle cleanup; update comments and the affected cleanup regression.
4. Run targeted survivor, count, and between-turn pruning tests plus changed-line lint/format checks and independent review; record evidence before Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Made `ConsoleAgentBridge.fleet_snapshot()` observational by removing its retained-owner cleanup call while preserving the existing service lookup and live survivor filtering.
- Extended the real bridge/controller regression to prove both live and terminal busy-count reads preserve retained owners and coordinator handles. The established `live_snapshot()` path still releases settled owners, and terminal coordinator handles remain until next-turn pruning.
- Updated lifecycle documentation and removed the controller's stale prune-on-read comment. No ADR was added; ADR-129 already defines the lifecycle contract.
- Targeted verification passes (10 tests). Ruff reports only existing large-file debt: 235 findings versus 237 at `HEAD`, zero findings on changed lines, and the same three files requiring whole-file formatting. `git diff --check` passes. Independent review and final task completion remain with the root task owner.
<!-- SECTION:NOTES:END -->
