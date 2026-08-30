---
id: TASK-24405
title: Add bounded Personal Context interview coordinator
status: Done
assignee:
  - '@codex'
created_date: '2026-08-30 00:51'
updated_date: '2026-08-30 02:25'
labels:
  - personal-context
  - interviews
  - privacy
dependencies:
  - TASK-24403
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide a reusable privacy-preserving twenty-question interview state machine whose reviewed output is the only path to canonical profile changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fixed interviews make no provider call and adaptive interviews pin one disclosed provider and model with at most twenty question attempts.
- [x] #2 Protected drafts are encrypted and local-only, support bounded resume and expiry, and destroy their draft key after commit, discard, or expiry.
- [x] #3 Finishing creates a deterministic review diff without mutating records or disclosing user-only profile content.
- [x] #4 Selected changes commit atomically through PersonalContextService and workspace interviews cannot write global records.
- [x] #5 Invalid questions, secret material, provider errors, and stale drafts fail safely.
- [x] #6 Targeted interview, privacy, service, and static verification passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED coordinator, draft-lifecycle, diff, privacy-owner, and atomic-commit tests covering fixed/adaptive modes, the 20-attempt ceiling, expiry, fallback, scope isolation, and no pre-review writes.
2. Implement fixed and configured-model question-provider boundaries against pinned Shared Core `tldw-profile-core==0.1.0` interview contracts, with tools disabled and strict single-question validation.
3. Add encrypted, unsynchronized interview drafts using the existing Personal Context cipher/key-protector boundary, including memory-only mode, resume/expiry, and key destruction.
4. Implement deterministic structured-key diffing and one service-owned selected-change transaction; keep raw answers and possible private duplicates outside logs, Sync, and canonical records.
5. Run targeted Personal Context/service regressions, privacy durable-owner checks, Ruff/format/diff checks, and independent specification/code-quality reviews before closeout.

ADR required: no

ADR path: `backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md`

Reason: ADR-102 already governs encrypted local interview drafts, final-review-only record creation, runtime authority, proposal separation, and evidence/privacy boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added fixed and configured-model interview providers plus a bounded coordinator with pinned provider/model and question-pack identity, strict topic/question validation, pre-provider scope authorization, session-wide attempt fencing, deterministic review diffs, and terminal post-commit recovery state.
- Added encrypted local-only draft persistence with authenticated expiry/revision metadata, optimistic concurrency, independent passphrase bundles, memory-only fallback, deep-copy isolation, and durable cleanup-pending recovery for key destruction across commit, discard, expiry, and partial failures.
- Added service/repository-owned atomic selected-change batches with manifest and record fences, workspace isolation, collision handling, Sync outbox writes for syncable records, and safe expired-record behavior. Registered the draft database in the private SQLite owner inventory.
- ADR check: existing [ADR-102](../decisions/102-personal-context-profile-authority-sync-and-encryption.md) remains the governing decision; no new ADR was required.
- Verification: `219 passed` for `Tests/Personal_Context` plus `Tests/DB/test_private_sqlite_inventory.py`; Ruff check and format check passed; `git diff --check` passed. Independent gates returned `SPEC APPROVED` and `CODE QUALITY APPROVED`. The existing Requests dependency warning and macOS pytest temporary-directory cleanup warnings remain unrelated.
- Implementation commit: `4429984101` (`feat: add bounded profile interview coordinator`).
<!-- SECTION:NOTES:END -->
