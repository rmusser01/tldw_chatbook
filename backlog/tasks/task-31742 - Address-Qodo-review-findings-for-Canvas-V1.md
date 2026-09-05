---
id: TASK-31742
title: Address Qodo review findings for Canvas V1
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 21:37'
updated_date: '2026-09-05 21:55'
labels:
  - canvas
  - review
dependencies:
  - TASK-31741
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve every finding posted on PR 2432 with verified corrections or evidence-backed explanations while preserving the approved V1 security and lifecycle contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All eight initial Qodo findings have a documented technical disposition and a reply in their original review thread.
- [x] #2 Queued Canvas card actions cannot resolve an old card against a different active conversation; current-card actions still work.
- [ ] #3 Valid review corrections preserve path authorization, transaction ownership, strict bounded wire validation, effective configuration precedence, compatibility and source-private diagnostics.
- [ ] #4 Targeted regression tests, independent review and required current-head CI support merge readiness; no security or performance gate is weakened.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR for direct review corrections. ADR path: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md; ADR-097 governs startup costs. Reason: preserve existing security and ownership boundaries; stop for design approval if a suggestion requires new authority or architecture. 1. Read every Qodo review body and inline comment; record stable comment IDs and evaluate against actual call paths and approved contracts. 2. Reproduce verified behavioral defects before changes, starting with stale card session routing; use one bounded correction at a time and retain first-use/strict-zero-egress coverage. 3. For path, transactions, bridge validation and configuration findings, use existing shared mechanisms only when semantics remain exact; document justified disagreement instead of inventing containment roots or loosening validation. 4. Correct public helper documentation, compatibility wrapper naming and bounded operational log context; audit diagnostic inventory before regeneration. 5. Run targeted checks and independent review, reply to every original thread with evidence, update the PR and wait for current-head protected CI and Qodo completion. Root exclusively executes isolated pytest/browser checks; no full sweep, OS resource changes or V2 work before merge.
<!-- SECTION:PLAN:END -->
