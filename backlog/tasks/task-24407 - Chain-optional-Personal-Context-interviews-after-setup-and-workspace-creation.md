---
id: TASK-24407
title: Chain optional Personal Context interviews after setup and workspace creation
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-30 05:43'
updated_date: '2026-08-30 05:55'
labels:
  - personal-context
  - interviews
  - onboarding
  - workspaces
dependencies:
  - TASK-24406
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Offer Personal Context interviews only after first-run setup or workspace creation has committed, while preserving the completed setup or workspace if the interview is skipped, cancelled, fails, or is unavailable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Completed first-run setup persists before an optional personal interview is launched and all existing exit-route continuations remain exactly once.
- [ ] #2 Workspace creation returns a fully committed workspace before an optional workspace interview is launched from every canonical caller.
- [ ] #3 Declining, cancelling, provider failure, and interview launch failure preserve the completed setup or workspace and continue safely.
- [ ] #4 First-run and workspace creation result contracts remain backward-compatible when the interview offer is false.
- [ ] #5 Production-shaped ordering tests and targeted existing wizard/workspace regressions pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing production-shaped tests for first-run completion ordering, rerun/exit-route behavior, all canonical workspace callers, cancellation/failure preservation, and exactly-once continuation.
2. Add one post-commit interview-launch helper that owns result normalization and continuation idempotency without owning setup persistence, registry mutation, or caller navigation.
3. Extend first-run and workspace result contracts with opt-in interview offers, then wire each caller only after its existing commit/finalize path succeeds.
4. Run targeted wizard/workspace and Personal Context regressions, inspect the complete diff, and obtain independent specification and code-quality review.

ADR required: no

ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md

Reason: ADR-102 already defines post-commit interview ownership and failure isolation; this task only connects the approved launch boundary to existing completion flows.
<!-- SECTION:PLAN:END -->
