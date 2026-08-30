---
id: TASK-24405
title: Add bounded Personal Context interview coordinator
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-30 00:51'
updated_date: '2026-08-30 00:52'
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
- [ ] #1 Fixed interviews make no provider call and adaptive interviews pin one disclosed provider and model with at most twenty question attempts.
- [ ] #2 Protected drafts are encrypted and local-only, support bounded resume and expiry, and destroy their draft key after commit, discard, or expiry.
- [ ] #3 Finishing creates a deterministic review diff without mutating records or disclosing user-only profile content.
- [ ] #4 Selected changes commit atomically through PersonalContextService and workspace interviews cannot write global records.
- [ ] #5 Invalid questions, secret material, provider errors, and stale drafts fail safely.
- [ ] #6 Targeted interview, privacy, service, and static verification passes.
<!-- AC:END -->

## Implementation Plan

1. Add RED coordinator, draft-lifecycle, diff, privacy-owner, and atomic-commit tests covering fixed/adaptive modes, the 20-attempt ceiling, expiry, fallback, scope isolation, and no pre-review writes.
2. Implement fixed and configured-model question-provider boundaries against pinned Shared Core `tldw-profile-core==0.1.0` interview contracts, with tools disabled and strict single-question validation.
3. Add encrypted, unsynchronized interview drafts using the existing Personal Context cipher/key-protector boundary, including memory-only mode, resume/expiry, and key destruction.
4. Implement deterministic structured-key diffing and one service-owned selected-change transaction; keep raw answers and possible private duplicates outside logs, Sync, and canonical records.
5. Run targeted Personal Context/service regressions, privacy durable-owner checks, Ruff/format/diff checks, and independent specification/code-quality reviews before closeout.

ADR required: no

ADR path: `backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md`

Reason: ADR-102 already governs encrypted local interview drafts, final-review-only record creation, runtime authority, proposal separation, and evidence/privacy boundaries.
