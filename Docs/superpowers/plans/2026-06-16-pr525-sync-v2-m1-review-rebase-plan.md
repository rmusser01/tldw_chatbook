# PR 525 Sync v2 M1 Review Rebase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebase PR #525 onto latest `dev` and resolve all actionable Sync v2 M1 review comments.

**Architecture:** Keep PR #525 scoped to M1 schema/transport conformance. Use model-level normalization for legacy capability input compatibility, centralize dry-run domain selection so coarse requests can match M1 dotted domains, and remove key-shaped literals from documentation examples without changing the live-check intent.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, Bandit, GitHub CLI, Backlog.md MCP.

**ADR required:** no new ADR

**ADR path:** `backlog/decisions/008-sync-v2-client-m1-contract-alignment.md`

**Reason:** ADR 008 already records the Sync v2 M1 contract decision; this task is bounded review remediation and rebase work against that accepted decision.

---

## Stage 1: Rebase Branch
**Goal:** Put `feat/sync-v2-m1-client-conformance` on top of current `origin/dev`.
**Success Criteria:** Branch rebases cleanly or conflicts are resolved without losing Sync v2 M1 behavior.
**Tests:** `git status --short --branch`, `git diff --check`.
**Status:** Complete

- [x] Fetch latest `origin/dev` and PR head.
- [x] Rebase the PR branch onto `origin/dev`.
- [x] Resolve any conflicts by preserving current `dev` plus scoped PR behavior.
- [x] Confirm `origin/dev` is an ancestor of `HEAD`.

## Stage 2: Review Comment Audit
**Goal:** Map each review comment to a code/test/doc change or an evidence-backed no-op.
**Success Criteria:** Qodo and Gemini threads are addressed in the rebased branch.
**Tests:** GitHub review-thread inspection plus local code inspection.
**Status:** Complete

- [x] Verify legacy `supported_operations` list input is normalized before Pydantic field validation.
- [x] Verify dry-run requested coarse domains match M1 dotted advertised domains.
- [x] Verify `domains=None` or missing legacy domains cannot crash dry-run capability filtering.
- [x] Remove hardcoded or key-shaped API key literals from the plan document.
- [x] Ignore CI check state per user instruction; only use local evidence.

## Stage 3: Regression Coverage
**Goal:** Add focused tests for the review-commented behavior before production fixes.
**Success Criteria:** Tests fail against the pre-fix behavior and pass after minimal implementation.
**Tests:** Focused `Tests/tldw_api` and `Tests/Sync_Interop` pytest targets.
**Status:** Complete

- [x] Add failing schema test for flat legacy `supported_operations`.
- [x] Add failing dry-run test for coarse request to dotted M1 domains.
- [x] Add failing dry-run test for nullable raw capability domains.
- [x] Implement minimal schema and service fixes.
- [x] Re-run focused tests.

## Stage 4: Verification And Finalization
**Goal:** Verify touched scope, update Backlog, commit, push, and resolve review threads.
**Success Criteria:** Local verification evidence is documented and PR branch is updated.
**Tests:** Focused pytest, `git diff --check`, Bandit on touched production paths.
**Status:** Complete

- [x] Run focused Sync tests.
- [x] Run Bandit on touched production paths and document any pre-existing findings.
- [x] Update `TASK-120` acceptance criteria and implementation notes.
- [x] Commit and push the rebased branch to PR #525.
- [x] Resolve addressed GitHub review threads.
