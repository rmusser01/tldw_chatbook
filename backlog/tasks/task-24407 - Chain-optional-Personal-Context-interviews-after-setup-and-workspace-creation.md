---
id: TASK-24407
title: Chain optional Personal Context interviews after setup and workspace creation
status: Done
assignee:
  - '@codex'
created_date: '2026-08-30 05:43'
updated_date: '2026-08-30 06:57'
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
- [x] #1 Completed first-run setup persists before an optional personal interview is launched and all existing exit-route continuations remain exactly once.
- [x] #2 Workspace creation returns a fully committed workspace before an optional workspace interview is launched from every canonical caller.
- [x] #3 Declining, cancelling, provider failure, and interview launch failure preserve the completed setup or workspace and continue safely.
- [x] #4 First-run and workspace creation result contracts remain backward-compatible when the interview offer is false.
- [x] #5 Production-shaped ordering tests and targeted existing wizard/workspace regressions pass.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added default-off interview offers to first-run setup and workspace creation while preserving the legacy result shape and continuation behavior when not selected.
- Routed setup, Console, Settings, and Library launches through one post-commit, exactly-once handoff. Setup/workspace state now survives cancellation, scope preparation failure, secure-draft fallback, adaptive-provider misconfiguration, launch failure, and notification failure.
- Added fixed personal/workspace question packs with namespaced semantic subjects, correct dislike polarity, and safe Goal/WorkingContext/Convention workspace records. Secure draft custody is capability-probed before durable use and falls back to disclosed memory-only drafts when unavailable.
- Kept partially created workspaces authoritative during folder-binding retries and froze the already-committed interview choice so Retry/Cancel cannot return a visually contradictory value.
- Verification: 145 primary targeted tests plus first-run callback (9), summary (11), full-track layout (2), and catalog completion (1) regressions passed; Ruff check, Ruff format check, and `git diff --check` passed. Independent reviews returned `SPEC APPROVED` and `CODE QUALITY APPROVED`.
- The full suite was not run under the repository's targeted-verification policy. `Tests/UI/test_console_new_workspace.py::test_console_new_workspace_creates_and_activates` remains an unrelated pre-modal rail-allocation failure (`Workspace section did not receive a usable allocation`); no changed handler or modal path participates in that allocation.
- ADR check: existing ADR-102 applies; no new architectural decision was introduced.
<!-- SECTION:NOTES:END -->
