---
id: TASK-1776
title: Add exact Console composer improvement transactions
status: To Do
assignee: []
created_date: '2026-08-01 23:30'
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
