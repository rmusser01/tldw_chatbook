---
id: TASK-1776
title: Add exact Console composer improvement transactions
status: In Progress
assignee: []
created_date: '2026-08-01 23:30'
updated_date: '2026-08-02 09:05'
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
- [ ] #1 The composer exposes immutable snapshot, model-facing projection, apply, restore, and invalidation behavior that preserves segment text, origin, labels, display state, cursor, selection, and ordering exactly.
- [ ] #2 Model-facing text omits inline-file content and metadata, uses collision-safe opaque placeholders, and rejects stale or tampered results before any mutation.
- [ ] #3 Apply and temporary Undo preserve pending attachments and exact prior state across success, no-change, cancellation, and stale-state cases, verified by focused transaction tests.
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
