---
id: TASK-31808
title: Reconcile detached and delegated fork census routes
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06'
updated_date: '2026-09-06 02:26'
labels:
  - tests
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The remaining fork census failure treats a pending-work policy field as a live
session mutation and omits a synchronous wrapper of an already guarded setter.
Recognize these cases without concealing future unguarded live mutations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The existing persistence drain is recognized as detached only with its exact local lifecycle and drain bindings; unexpected rebindings or lifecycle poisoning remain detectable.
- [x] #2 Direct live writes and new mutating callees remain visible through all three persistence roots.
- [x] #3 The committed name setter is classified only with the exact guarded delegation, no local fork writes and no additional mutating child.
- [x] #4 Complete census and affected behavioral files pass, negative mutations exercise the actual classifier, and scoped static checks and independent review qualify the test-only repair.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add positive real-method and negative AST mutation tests before changing the classifier. Exercise carrier rebinding, lifecycle poisoning, live writes, delegated child mutations and wrong name-setter receiver/arguments.
2. Narrow only the pending carrier assignment event after checking exact lifecycle construction and all local drain bindings, including None. Do not exempt the method or bypass recursive call traversal. Register the committed name setter and strengthen its delegated-route assertion.
3. Run the complete census and affected settings/fork/publication/lifetime files. Run scoped Ruff/format, obtain independent read-only review, record evidence and remaining broader failures, and update the existing draft PR.

ADR required: no
ADR path: backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md
Reason: Test-only correction implementing the existing distinction between immutable fork source configuration and excluded pending runtime work; no runtime or authority changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Test-only fork census repair: recognize only the pending-drain policy assignment after checking exact local lifecycle/drain provenance; preserve recursive live-write and callee analysis. Register the committed name wrapper with exact guarded receiver/argument checks and no additional mutations. Initial complete census: 8 failed/36 passed, proving detached false-positive and six delegation bypasses. First correction: 44 passed. Independent review exposed alternative aliases and pattern capture; four added controls failed before correction, then passed. Final six complete census/settings/publication/first-send/fork/lifetime files: 351 passed in 48.45 seconds, two existing dependency warnings; /private/tmp/tldw-census-classification-final.xml. Census alone: 48 passed. Scoped Ruff/format/diff and independent re-review pass. No production code, diagnostic pins, broad exemptions or direct-owner inventory changes. Added the binding-syntax lesson and updated the checkpoint; broader failures remain open. ADR check: existing ADR-092, no new decision required.
<!-- SECTION:NOTES:END -->
