---
id: TASK-643
title: Make runtime policy the sole application runtime-source authority
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-26 13:35'
updated_date: '2026-07-26 15:02'
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
Reason: ADR-026 defines the runtime authority, private persistence, mutation, projection, and thread-affinity contract.
Full plan: Docs/superpowers/plans/2026-07-26-task-643-runtime-policy-authority.md

1. Move runtime-policy JSON to the effective config path and ADR-022 private-file primitives.
2. Replace mutable context state with owner-thread-affine revision snapshots and persist-before-publish commits.
3. Make capability refresh discard superseded probes and side effects.
4. Remove TldwCli AppState dependency, delete the projection boundary's AppState mirror, and remove all independent runtime projection writers.
5. Add ownership/privacy guards and run focused gates, but keep TASK-643 In Progress until the shared TASK-646 release gates.
<!-- SECTION:PLAN:END -->
