# PR 526 Sync v2 P2 Review Rebase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebase PR #526 onto latest `dev` and resolve all actionable Sync v2 P2 `notes.note` review comments.

**Architecture:** Keep PR #526 scoped to the P2 `notes.note` vertical. Preserve the additive M1 superset approach from ADR 008 and the P2 plan while hardening external-input boundaries, mirror persistence, idempotent acknowledgements, and legacy encrypted apply behavior.

**Tech Stack:** Python 3.11+, Pydantic v2, SQLite, pytest, Bandit, GitHub CLI, Backlog.md.

**ADR required:** no

**ADR path:** `backlog/decisions/008-sync-v2-client-m1-contract-alignment.md`

**Reason:** ADR 008 already records the Sync v2 M1 contract and P2 review remediation does not change the accepted architecture.

---

## Stage 1: Rebase Branch
**Goal:** Put PR #526 on top of current `origin/dev`.
**Success Criteria:** Branch rebases cleanly or conflicts are resolved while preserving P2 notes behavior.
**Tests:** `git status --short --branch`, `git merge-base --is-ancestor origin/dev HEAD`, `git diff --check`.
**Status:** Complete

- [x] Fetch latest `origin/dev` and PR head.
- [x] Rebase worktree branch onto `origin/dev`.
- [x] Resolve conflicts without losing PR #525 fixes already on `dev`.
- [x] Confirm `origin/dev` is an ancestor of `HEAD`.

## Stage 2: Review Comment Audit
**Goal:** Map every unresolved PR thread to a scoped fix or evidence-backed no-op.
**Success Criteria:** Gemini and Qodo review threads are addressed by code, tests, or doc changes.
**Tests:** GitHub review-thread inspection plus local code inspection.
**Status:** Complete

- [x] Remove hardcoded/key-shaped API key examples from the P2 plan.
- [x] Guard `NotesM1SyncAdapter` required dependencies and identifiers.
- [x] Validate and sanitize server-provided note payload before persistence.
- [x] Harden `NotesMirror` path handling and transaction usage.
- [x] Update mirror handling for accepted and idempotent push acknowledgements.
- [x] Guard legacy apply paths when `dataset_key` is absent.
- [x] Reject Sync v2 envelopes missing both `entity_id` and `object_id`.
- [x] Add Google-style docstring details to `canonical_payload_hash`.
- [x] Ignore CI check state per user instruction; only local evidence is used.

## Stage 3: Regression Coverage
**Goal:** Prove review-commented behavior fails before implementation and passes after fixes.
**Success Criteria:** Focused tests cover the corrected boundary and persistence behavior.
**Tests:** Focused `Tests/Sync_Interop` and `Tests/tldw_api` pytest targets.
**Status:** Complete

- [x] Add failing tests for missing mirror/dataset/id, note payload validation/sanitization, mirror path/transactions, idempotent mirror updates, missing dataset key, and missing envelope identifiers.
- [x] Implement minimal fixes matching existing Sync_Interop patterns.
- [x] Re-run the focused tests to green.

## Stage 4: Verification And Finalization
**Goal:** Verify touched scope, update Backlog, commit, push, and resolve review threads.
**Success Criteria:** Local verification evidence is documented and PR #526 branch is updated.
**Tests:** Focused pytest, `git diff --check`, Bandit on touched production paths.
**Status:** Complete

- [x] Run focused Sync v2 tests.
- [x] Run `git diff --check`.
- [x] Run Bandit on touched production paths and document any pre-existing findings.
- [x] Update `TASK-121` acceptance criteria and implementation notes.
- [x] Commit and push the rebased branch back to PR #526.
- [x] Resolve addressed GitHub review threads.
