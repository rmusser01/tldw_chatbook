---
id: TASK-6.3
title: Phase 5.3 Apply recovery taxonomy to runtime-policy blockers
status: Done
assignee:
  - '@codex'
created_date: '2026-05-05 03:40'
labels:
  - unified-shell
  - phase-5
  - recovery
  - runtime-policy
dependencies: []
documentation:
  - >-
    Docs/superpowers/qa/unified-shell/phase-5/2026-05-05-shared-recovery-taxonomy.md
  - >-
    Docs/superpowers/qa/unified-shell/phase-4/2026-05-05-phase-4-destination-service-adoption-closeout.md
parent_task_id: TASK-6
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Apply the shared Phase 5 recovery taxonomy to visible runtime-policy denial states in service-backed shell destinations so users can distinguish wrong-source, server setup/auth/session, policy, and disabled-capability blockers from generic service failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Skills, Library, Personas, and W+C destination policy-denied states expose what is unavailable, why, next action, recovery target, authority owner, stable selector, and disabled tooltip copy.
- [x] #2 Runtime-policy reason codes are mapped to user-facing recovery states for wrong source, server setup, server auth/session, policy denied, and capability disabled blockers.
- [x] #3 Automated UI regressions verify visible recovery copy and disabled tooltips for representative destination policy denials.
- [x] #4 Durable Phase 5 QA evidence records verification, residual risks, and the remaining Phase 5 blocker families.
- [x] #5 Parent Phase 5 tracking is updated without marking the phase verified.
<!-- AC:END -->

## Implementation Plan

1. Add failing UI regressions for representative policy-denied states across Skills, Library, Personas, and W+C destination shells.
2. Add the smallest shared recovery helper needed to map `PolicyDeniedError` reason codes into `DestinationRecoveryState` fields.
3. Wire the helper into the four service-backed destination shells while preserving existing service-error and empty-state behavior.
4. Update Phase 5 QA evidence, roadmap, and Backlog tracking for `TASK-6.3`.
5. Run focused UI and Phase 5 tracking tests before opening the PR.

## Implementation Notes

Added a shared runtime-policy recovery mapper that converts `PolicyDeniedError.reason_code` values into `DestinationRecoveryState` copy and tooltips. Wired it into Skills, Library, Personas, and W+C service-backed destination blocked states while preserving generic service-error and empty-state behavior. Added focused UI regressions for authority-denied, wrong-source, server-auth, and server-session blockers, plus Phase 5 tracking evidence for `TASK-6.3`.
