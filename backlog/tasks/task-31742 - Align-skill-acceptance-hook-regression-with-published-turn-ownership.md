---
id: TASK-31742
title: Align skill acceptance-hook regression with published turn ownership
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:49'
updated_date: '2026-09-05 19:53'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve the successful and refused skill-send hook contracts while recognizing the pending assistant owner published before acceptance under current Console admission.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The success hook fires exactly once after skill substitution and pending-owner publication but before any provider call; the same pending owner completes afterward.
- [x] #2 Refused skills still invoke no accepted hook and create no assistant placeholder.
- [x] #3 The complete skill-substitution file and scoped static checks pass with no runtime behavior change.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the existing before-assistant assertion failure and trace current ephemeral and durable acceptance publication.
2. Rename the stale regression and snapshot immutable pending-owner fields, completed skill execution, and provider-call count at the hook.
3. Assert exact hook cardinality, pending empty owner, zero provider calls at hook time, and same-owner completion; retain refusal coverage unchanged.
4. Run the complete skill-substitution file and scoped static checks; independently review before commit.
ADR required: no
ADR path: backlog/decisions/079-console-library-conversation-authority.md
Reason: Test-only reconciliation with existing recovery-owner publication; no new storage, runtime, or UI contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aligned the acceptance hook regression with the already-published pending assistant owner, while asserting zero provider calls at the hook and completion of that exact owner afterward. Refusal coverage remains unchanged. ADR check: no new ADR; preserves ADR-079/090 admission and ownership. Before: the two whole affected files had 62 passes and the two diagnosed failures. After both independent fixture corrections: 64 passed in 18.72s (/private/tmp/tldw-31741-31742-final.xml). Whole-file Ruff lint/format and whitespace checks passed; parent reviewed immutable hook snapshots without findings.
<!-- SECTION:NOTES:END -->
