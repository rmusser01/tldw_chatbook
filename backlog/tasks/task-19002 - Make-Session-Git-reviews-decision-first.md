---
id: TASK-19002
title: Make Session Git reviews decision-first
status: Done
assignee:
  - '@codex'
created_date: '2026-08-20 07:40'
updated_date: '2026-08-20 21:57'
labels:
  - notes
  - git
  - ux
  - accessibility
dependencies:
  - TASK-19001
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Lead commit and push reviews with the facts required for authorization while retaining implementation and recovery evidence through progressive disclosure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Commit and push reviews lead with `What`, `Where`, `Impact`, and `Recovery` sections.
- [x] #2 Every fact that can change authorization remains visible without opening technical details, including exact destination, candidate, lease, hooks, transport, and publication scope.
- [x] #3 Technical details are collapsed by default, contain audit-only evidence, and remain keyboard-operable with correct focus restoration.
- [x] #4 Existing Git trust, staging, commit, push, uncertainty, and cancellation contracts remain unchanged.
- [x] #5 The decision facts and disclosure remain contained, scrollable, and keyboard-safe at 40x20.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED commit/push review tests requiring visible What, Where, Impact, and Recovery sections plus every authorization-changing destination/candidate/lease/hooks/transport/publication fact outside technical details.\n2. Reshape only the existing sanitized review projections and panel composition using already-owned immutable facts; preserve domain services, trust, staging, commit, push, cancellation, and uncertainty policy.\n3. Restrict the collapsed technical disclosure to audit-only evidence, keep Endpoint Details independently reachable, and prove focus restoration plus 40x20 scrolling/keyboard safety.\n4. Run commit/push suites, cumulative presentation gates, static checks, spec/quality review, documentation updates, and close with exact evidence.\n\nADR required: no new ADR\nADR path: backlog/decisions/035-file-notes-session-git-index-controls.md; backlog/decisions/038-file-notes-guarded-session-commit.md; backlog/decisions/039-file-notes-guarded-session-push.md\nReason: presentation-only reorganization of existing immutable Git authorization evidence; execution and authority policies remain unchanged.\n\nPlan: Docs/superpowers/plans/2026-08-20-notes-files-presentation-refinement.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented decision-first Session Git reviews without changing Git trust, staging, commit, push, cancellation, uncertainty, or domain-service behavior. Commit and push reviews now lead with visible What, Where, Impact, and Recovery sections; exact repository/destination, branch/ref/endpoint, candidate transition, lease, hook/signing, transport/authentication, publication scope, and recovery facts remain outside collapsed Technical details. Technical details default collapsed and contain audit-only repository identity/refspec evidence; Endpoint details remains independent with focus restoration. Sanitized panel projections carry already-owned immutable RepositoryIdentity facts. Fresh owner snapshot candidate equality now gates local-proof authorization, blocked retry, and preflight review admission, so trust ABA cannot revive stale authority; expired reviews use a functional Back to session recovery. The File Notes guide and 40x20 keyboard/compositor coverage were updated. Existing ADR-035, ADR-038, and ADR-039 govern unchanged execution policy; no new ADR was required. Verification: full Git UI suites 212 passed in independent quality review; implementation gates reached 210/64 as regressions were added; root final slice 12 passed; Ruff, CSS bundle parity, and git diff --check passed. Independent spec and quality reviews approved with no remaining findings.
<!-- SECTION:NOTES:END -->
