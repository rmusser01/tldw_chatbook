---
id: TASK-643
title: Make runtime policy the sole application runtime-source authority
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-26 13:35'
updated_date: '2026-07-26 18:44'
labels:
  - architecture
  - state
  - reliability
dependencies: []
references:
  - backlog/decisions/026-application-session-state-ownership.md
documentation:
  - >-
    Docs/superpowers/specs/2026-07-26-application-session-state-ownership-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the active application dependency on the misleading root AppState model and make runtime-source publication durable, revisioned, and resistant to stale asynchronous capability results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 TldwCli no longer imports or instantiates AppState while the exported legacy state models remain importable and retain their documented serialization behavior
- [ ] #2 RuntimePolicyContext exposes read-only state plus creating-thread-affine revision snapshots and persist-before-publish compare-and-swap commits whose persistence failures leave the prior state unchanged, with no direct setter or persistence escape hatch
- [ ] #3 Stale asynchronous capability refresh results cannot overwrite a newer runtime source or server identity
- [ ] #4 All production runtime-source and capability writers use the authoritative context and app-level compatibility projections have no independent writers
- [ ] #5 Runtime-policy persistence follows the effective config path, uses the ADR-022 private read/atomic-write boundary, fails closed for unsafe POSIX targets, reports Windows privacy posture as unverified, and neither falls back to nor migrates from the ordinary default path under an override
- [ ] #6 Focused runtime-policy, off-owner mutation, persistence-failure recovery, privacy-sentinel, scoped static, ownership-guard, and architecture checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/026-application-session-state-ownership.md
Reason: ADR-026 defines the runtime authority, private persistence, mutation, projection, thread-affinity, bootstrap, and coordinated configuration-rebind contract.
Full corrective plan: Docs/superpowers/plans/2026-07-26-task-643-structural-ownership-correction.md
Original partially superseded plan: Docs/superpowers/plans/2026-07-26-task-643-runtime-policy-authority.md
Structural specification: Docs/superpowers/specs/2026-07-26-task-643-structural-ownership-enforcement-design.md

1. Replace application-shaped bootstrap tests with direct app-independent units or full TldwCli coverage.
2. Enforce getter-only atomic runtime projections and exact publication boundaries.
3. Make context internals private and installation failure-atomic and one-time.
4. Add a post-commit provider rebind primitive with bounded cleanup/materialization failures.
5. Coordinate commit-first runtime changes through the real application and adapt the Schedules caller structurally.
6. Rebind the actual mounted Settings screen without replacing runtime authority.
7. Run direct/full-application, privacy-sentinel, structural, and scoped static gates while keeping TASK-643 In Progress until TASK-646 shared release gates.
<!-- SECTION:PLAN:END -->
