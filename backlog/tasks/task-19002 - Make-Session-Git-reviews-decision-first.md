---
id: TASK-19002
title: Make Session Git reviews decision-first
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-20 07:40'
updated_date: '2026-08-20 21:03'
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
- [ ] #1 Commit and push reviews lead with `What`, `Where`, `Impact`, and `Recovery` sections.
- [ ] #2 Every fact that can change authorization remains visible without opening technical details, including exact destination, candidate, lease, hooks, transport, and publication scope.
- [ ] #3 Technical details are collapsed by default, contain audit-only evidence, and remain keyboard-operable with correct focus restoration.
- [ ] #4 Existing Git trust, staging, commit, push, uncertainty, and cancellation contracts remain unchanged.
- [ ] #5 The decision facts and disclosure remain contained, scrollable, and keyboard-safe at 40x20.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED commit/push review tests requiring visible What, Where, Impact, and Recovery sections plus every authorization-changing destination/candidate/lease/hooks/transport/publication fact outside technical details.\n2. Reshape only the existing sanitized review projections and panel composition using already-owned immutable facts; preserve domain services, trust, staging, commit, push, cancellation, and uncertainty policy.\n3. Restrict the collapsed technical disclosure to audit-only evidence, keep Endpoint Details independently reachable, and prove focus restoration plus 40x20 scrolling/keyboard safety.\n4. Run commit/push suites, cumulative presentation gates, static checks, spec/quality review, documentation updates, and close with exact evidence.\n\nADR required: no new ADR\nADR path: backlog/decisions/035-file-notes-session-git-index-controls.md; backlog/decisions/038-file-notes-guarded-session-commit.md; backlog/decisions/039-file-notes-guarded-session-push.md\nReason: presentation-only reorganization of existing immutable Git authorization evidence; execution and authority policies remain unchanged.\n\nPlan: Docs/superpowers/plans/2026-08-20-notes-files-presentation-refinement.md
<!-- SECTION:PLAN:END -->
