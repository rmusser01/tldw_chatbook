---
id: TASK-24723
title: Complete Personal Context proposal review and privacy evidence
status: Done
assignee:
  - '@codex'
created_date: '2026-08-30 15:28'
updated_date: '2026-08-30 16:15'
labels:
  - personal-context
  - settings
  - privacy
  - docs
dependencies:
  - TASK-24408
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users a clear Settings workflow to inspect and resolve agent-proposed profile changes, prove rejected content is removed from every default durable owner, and document the privacy and authority model.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings lists pending proposals without exposing user-only records and supports accept, edit-and-accept, and reject through the proposal service.
- [x] #2 Proposal review clearly identifies agent provenance, scope, operation, and possible private-duplicate risk before any mutation.
- [x] #3 Irreversible proposal resolution cannot be dismissed mid-commit and reports success, conflict, expiry, and recovery-safe failure states.
- [x] #4 Every terminal proposal becomes a content-free receipt across default durable owners; accepted content survives only in the canonical record and its required sync outbox.
- [x] #5 User documentation explains interviews, proposal review, direct-write evidence, agent permissions, private records, scope behavior, deletion, and shared Chatbook/server sync semantics.
- [x] #6 Production-shaped UI, durable-owner, proposal lifecycle, and relevant Plan 02 regressions pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing production-shaped Settings proposal-review tests and a durable-owner privacy inventory that exercises the real provider and review seams.
2. Extend the existing Personal Context review modal and Settings panel with pending-proposal actions, explicit provenance/scope warnings, edit-and-accept, and dismissal-safe commit states.
3. Verify content shredding across the profile DB/WAL, outbox, logs, diagnostics, caches, exports, and Console run persistence reached by the real path.
4. Document the full user workflow and authority/privacy model, then run targeted Plan 02 regression, lint, format, and privacy checks and obtain independent review.

ADR required: no

ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md

Reason: ADR-102 already defines proposal ownership, terminal content shredding, agent authority, record privacy, and shared sync semantics; this task completes the accepted review surface and evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added the Settings proposal queue and review modal with bounded provenance, scope, operation, private-duplicate warnings, exact visible-target previews, edit-and-accept, reject, and dismissal-safe outcome handling.
- Extended proposal acceptance to validate user-reviewed rewrites transactionally and refresh approved working-context retention, while preserving content-free terminal receipts.
- Enabled SQLite secure deletion and non-blocking WAL truncation after privacy-sensitive commits and application-owned export snapshots. Exact-byte durable-owner tests cover old proposal ciphertext and wrapped DEKs without disabling WAL concurrency.
- Documented interviews, profile scopes, privacy, agent permissions, deletion, proposal review, and the negotiated Chatbook/server synchronization contract. Recorded the WAL privacy/concurrency incident in the testing-evidence lessons.
- Reused ADR-102; no new architecture decision was required. Independent code-quality re-review approved the final WAL-preserving design.
- Verification: 263 targeted tests passed, including repository/export, proposal lifecycle, durable-owner inventory, production-shaped Settings/interview UI, agent tools, workspace handoff, and CSS integrity. Ruff check, Ruff format check, Python compilation, CSS generation, and `git diff --check` passed. The full suite was not run, per the repository's targeted-verification policy.
<!-- SECTION:NOTES:END -->
