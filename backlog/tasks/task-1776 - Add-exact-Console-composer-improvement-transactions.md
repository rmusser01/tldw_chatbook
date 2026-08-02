---
id: TASK-1776
title: Add exact Console composer improvement transactions
status: Done
assignee: []
created_date: '2026-08-01 23:30'
updated_date: '2026-08-02 10:19'
labels: []
dependencies:
  - TASK-1775
references:
  - Docs/superpowers/plans/2026-08-01-console-prompt-improvement-workbench.md
  - Docs/superpowers/specs/2026-08-01-console-prompt-improvement-design.md
  - >-
    backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give improvement flows a public, exact mutation boundary over the Console composer’s segment model. This stage prevents improvements from leaking inline-file data or flattening protected segments, and makes Apply and temporary Undo reliable before provider-backed UX is introduced.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The composer exposes immutable snapshot, model-facing projection, apply, restore, and invalidation behavior that preserves segment text, origin, labels, display state, cursor, selection, and ordering exactly.
- [x] #2 Model-facing text omits inline-file content and metadata, uses collision-safe opaque placeholders, and rejects stale or tampered results before any mutation.
- [x] #3 Apply and temporary Undo preserve pending attachments and exact prior state across success, no-change, cancellation, and stale-state cases, verified by focused transaction tests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing snapshot/origin and same-text generation tests for typed, pasted, inline-file, selection, cursor, stash/load, and attachment isolation.
2. Add failing projection and atomic apply/veto tests for collision-safe opaque placeholders, no inline-file leakage, exact rehydration, staleness, and malformed/tampered inputs.
3. Add failing temporary improvement Undo and invalidation tests for manual edit, send/stash, load/session replacement, later Apply, failure, and no-change.
4. Implement the minimal ADR-040 composer-owned immutable snapshot/projection/apply/restore boundary with explicit segment origins and one-swap mutation semantics.
5. Prove placeholder cardinality/order and edit-serial/generation guards are mutation-sensitive; run focused Ruff/format, diff checks, the exact gate, and relevant modal/native composer regressions.

ADR required: yes
ADR path: backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
Reason: ADR-040 already governs immutable snapshots, protected opaque projection, atomic Apply, and exact Undo; no new ADR is needed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Post-commit review status: CHANGES_REQUIRED. The first review found two Task 9 boundary defects: user-authored reserved placeholder syntax could project successfully and then be rejected on unchanged Apply, and attachment/image-only empty send-stash invocations did not expire improvement Undo. The correction rejects the complete reserved placeholder namespace in literal/paste model-facing source before token generation, preserves safe Unicode near-spellings byte-for-byte, and invalidates improvement Undo at every stash/send invocation before the empty-text return without mutating pending attachment identity or state. TDD correction evidence: 5 expected RED failures plus 1 safe-near-spelling pass; corrected slice 6 passed; full transaction suite 45 passed; broad composer suite 226 passed; full internals suite 127 passed. Existing ADR-040 still governs the boundary. TASK-1776 remains In Progress with all acceptance criteria unchecked pending independent re-review.

Stage closeout: exact composer transaction implementation and correction are complete and independently approved. Final implementation commits are 6432da961 and b19d43d00. Verification: exact Task 9 gate 386 passed; final transaction suite 45 passed; broad composer correction gate 226 passed; internals 127 passed; Ruff, py_compile, formatting checks for changed focused files, and diff checks clean. Mutation sensitivity proved placeholder cardinality/order/missing checks and edit-serial/generation checks are required, with guards restored. Final review confirmed reserved placeholder source collisions fail at projection, safe Unicode near-spellings preserve byte-identical no-change, every send/stash boundary expires Undo including empty and attachment-only cases, pending attachments remain unchanged, and cross-composer/cross-snapshot replay fails. ADR-040 remains governing. Task 10 progression authorized.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added an immutable, reversible Console composer transaction boundary with explicit segment origins, protected model projection, atomic stale/tamper checks, exact restore, and temporary Undo invalidation.
<!-- SECTION:FINAL_SUMMARY:END -->
