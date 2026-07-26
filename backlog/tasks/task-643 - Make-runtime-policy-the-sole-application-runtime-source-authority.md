---
id: TASK-643
title: Make runtime policy the sole application runtime-source authority
status: To Do
assignee:
  - '@codex'
created_date: '2026-07-26 13:35'
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
